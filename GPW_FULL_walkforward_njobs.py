# -*- coding: utf-8 -*-
"""
GPW WIG140 Backtest + AI + Fundamenty — FULL (walk-forward)
- Ceny: Stooq (spółki + WIG20)
- Strategia bazowa: MFI/RSI (grid)
- Cechy AI: techniczne, zmienność/ATR/HV, relacje do WIG20, fundamenty (Yahoo)
- Balans klas: SMOTEENN (jeśli imblearn dostępny), inaczej class_weight="balanced"
- Model: RandomForest (n_jobs=-1)
- Raporty: wyniki_*.xlsx, model_stats_*.xlsx, model_plots_*.pdf, model_summary_*.pdf, sygnaly_*_ai.xlsx
"""
import os, time, random, itertools, warnings, datetime as dt
from io import StringIO
import numpy as np, pandas as pd, requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm import tqdm
from ta.volume import MFIIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf

IMBLEARN_OK = True
try:
    from imblearn.combine import SMOTEENN
except Exception:
    IMBLEARN_OK = False

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        def deco(f): return f
        return deco

# ===== CONFIG =====
INITIAL_CAPITAL = 10_000.0
LOOKBACK_DAYS = 90
INDICATOR_WINDOW = 14
AI_TRAIN_DAYS = 240
AI_TEST_DAYS  = 15
AI_STEP_DAYS  = 7
FORWARD_HORIZON   = 10
TARGET_THRESHOLD  = 0.0
AI_PROBA_THRESHOLD = 0.55
RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH    = None
CLASS_WEIGHT = "balanced"
TOP_N = 10
CACHE_DIR = "cache_gpwwig"; os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_DAYS = 1
FUND_TTL_DAYS  = 7
MFI_LOWER_RANGE = [15, 20, 25, 30, 35, 40]
MFI_UPPER_RANGE = [65, 70, 75, 80]
RSI_LOWER_RANGE = [20, 30, 40, 50]
RSI_UPPER_RANGE = [60, 70, 80]
STOOQ_URLS = ["https://stooq.com/q/d/l/?s={sym}&i=d", "https://stooq.pl/q/d/l/?s={sym}&i=d"]
WIG20_SYM = "wig20"

def _requests_session():
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s

def cache_path(sym, kind="stooq"):
    sym_clean = sym.replace(":","_").replace("/","_")
    return os.path.join(CACHE_DIR, f"{kind}_{sym_clean}.csv")

def cache_is_fresh_days(path, days=1):
    if not os.path.exists(path): return False
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
    return (dt.datetime.now() - mtime).days < days

def fetch_bankier_tickers():
    url = "https://www.bankier.pl/inwestowanie/profile/quote.html?symbol=WIG140"
    try:
        s = _requests_session(); r = s.get(url, timeout=10); r.raise_for_status()
    except Exception as e:
        print("Błąd pobierania listy:", e); return []
    soup = BeautifulSoup(r.text, "html.parser")
    candidate = None
    for table in soup.find_all("table"):
        txt = table.get_text(" ", strip=True).lower()
        if any(k in txt for k in ("ticker","symbol","skrót","nazwa")): candidate = table; break
    if candidate is None: return []
    tks = []
    for tr in candidate.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds)>=2:
            tk = tds[1].get_text(strip=True)
            if tk: tks.append(tk if tk.endswith(".WA") else f"{tk}.WA")
    return list(dict.fromkeys(tks))

STQ_PL2EN = {"Data":"Date","Otwarcie":"Open","Najwyzszy":"High","Najwyższy":"High","Najnizszy":"Low","Najniższy":"Low","Zamkniecie":"Close","Zamknięcie":"Close","Wolumen":"Volume"}
def normalize_stooq_headers(df):
    df = df.copy(); df.columns = [STQ_PL2EN.get(c,c) for c in df.columns]; return df

def fetch_from_stooq(ticker):
    sym = ticker.replace(".WA","").lower(); urls=[u.format(sym=sym) for u in STOOQ_URLS]
    p = cache_path(sym,"stooq")
    if cache_is_fresh_days(p, CACHE_TTL_DAYS):
        try: return pd.read_csv(p)
        except Exception: pass
    for url in urls:
        try:
            s=_requests_session(); r=s.get(url,timeout=10); r.raise_for_status()
            df=pd.read_csv(StringIO(r.text)); 
            if df is None or df.empty: continue
            df=normalize_stooq_headers(df); df.to_csv(p,index=False); return df
        except Exception: continue
    return None

def prepare_stooq_df(df, start_dt, end_dt):
    if df is None: return None
    df=normalize_stooq_headers(df.copy())
    if "Date" not in df.columns: return None
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce"); df=df.dropna(subset=["Date"])
    df=df[(df["Date"]>=start_dt)&(df["Date"]<=end_dt)]
    if df.empty: return None
    df.rename(columns={"High":"HIGH","Low":"LOW","Close":"CLOSE","Volume":"VOL"}, inplace=True)
    if "VOL" not in df.columns: df["VOL"]=0
    df["VOL"]=df["VOL"].fillna(0)
    need=("HIGH","LOW","CLOSE","VOL")
    if not all(c in df.columns for c in need): return None
    return df.set_index("Date").sort_index()

def replace_inf_with_nan(df):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c]=df[c].replace([np.inf,-np.inf],np.nan)
    return df

def clip_extremes(df):
    bounds={"ATR_ratio":(0,1.5),"BBWidth20":(0,3),"HV20":(0,3),"PE_TTM":(0,100),"PE_FWD":(0,100),
            "P_B":(0,30),"EV_EBITDA":(-200,200),"ROE":(-1,1),"ROA":(-1,1),"D_E":(0,20),
            "DIV_YIELD":(0,0.3),"REV_GROWTH":(-1,1),"EPS_GROWTH":(-1,1)}
    for col,(lo,hi) in bounds.items():
        if col in df.columns: df[col]=df[col].clip(lower=lo,upper=hi)
    return df

def add_volatility_features(d):
    d=d.copy(); prev_close=d["CLOSE"].shift(1)
    tr = pd.concat([d["HIGH"]-d["LOW"], (d["HIGH"]-prev_close).abs(), (d["LOW"]-prev_close).abs()],axis=1).max(axis=1)
    d["ATR14"]=tr.rolling(14,min_periods=7).mean(); d["ATR_ratio"]=d["ATR14"]/d["CLOSE"]
    d["RET_D"]=d["CLOSE"].pct_change(); d["Volatility14"]=d["RET_D"].rolling(14,min_periods=7).std()
    sma20=d["CLOSE"].rolling(20,min_periods=10).mean(); std20=d["CLOSE"].rolling(20,min_periods=10).std()
    bbwidth=(4*std20)/sma20; bbwidth[(sma20.abs()<1e-12)]=np.nan; d["BBWidth20"]=bbwidth
    logret=np.log(d["CLOSE"]).diff(); d["HV20"]=logret.rolling(20,min_periods=10).std()*np.sqrt(252.0)
    return d

def add_indicators(df, window=INDICATOR_WINDOW):
    df=df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["MFI"]=MFIIndicator(high=df["HIGH"],low=df["LOW"],close=df["CLOSE"],volume=df["VOL"],window=window).money_flow_index()
        df["RSI"]=RSIIndicator(close=df["CLOSE"],window=window).rsi()
    return df.dropna()

@njit(cache=True, fastmath=True)
def simulate_strategy(close_prices, mfi, rsi, mfi_lower, mfi_upper, rsi_lower, rsi_upper, initial_capital):
    n=len(close_prices); signals=np.zeros(n, dtype=np.int8)
    capital=initial_capital; position=0; shares=0.0
    for i in range(1,n):
        cp=close_prices[i]; current_mfi=mfi[i]; current_rsi=rsi[i]
        if position==1 and (current_mfi>mfi_upper or current_rsi>rsi_upper):
            capital=shares*cp; shares=0.0; position=0; signals[i]=-1; continue
        if position==0 and (current_mfi<mfi_lower and current_rsi<rsi_lower):
            position=1; shares=capital/cp; signals[i]=1
    if position==1: capital=shares*close_prices[-1]
    return capital, signals

def backtest_arrays(close_arr, mfi_arr, rsi_arr, mfi_lower, mfi_upper, rsi_lower, rsi_upper, initial_capital):
    final_capital, signals = simulate_strategy(close_arr, mfi_arr, rsi_arr, mfi_lower, mfi_upper, rsi_lower, rsi_upper, initial_capital)
    return float(final_capital), signals

def process_ticker(ticker, df_raw):
    if df_raw is None or df_raw.empty: return None
    df=add_indicators(df_raw); 
    if df.empty: return None
    close_values=df["CLOSE"].to_numpy(np.float64); mfi_values=df["MFI"].to_numpy(np.float64); rsi_values=df["RSI"].to_numpy(np.float64)
    best_cap=-np.inf; best_params=None; best_signals=None; best_index=None
    for mfi_low,mfi_up,rsi_low,rsi_up in itertools.product(MFI_LOWER_RANGE,MFI_UPPER_RANGE,RSI_LOWER_RANGE,RSI_UPPER_RANGE):
        if mfi_low>=mfi_up or rsi_low>=rsi_up: continue
        cap, sig = backtest_arrays(close_values,mfi_values,rsi_values,mfi_low,mfi_up,rsi_low,rsi_up,INITIAL_CAPITAL)
        if cap>best_cap:
            best_cap=cap; best_params={"mfi_lower":mfi_low,"mfi_upper":mfi_up,"rsi_lower":rsi_low,"rsi_upper":rsi_up}
            best_signals=sig.copy(); best_index=df.index
    if best_params is None: return None
    sig_df=pd.DataFrame({"Date":best_index[1:], "Signal":best_signals[1:].astype(np.int8)}); sig_df=sig_df[sig_df["Signal"]!=0].copy(); sig_df["Ticker"]=ticker
    percent_return=(best_cap/INITIAL_CAPITAL-1.0)*100.0
    return {"Ticker":ticker, **best_params, "Final_Capital":round(best_cap,2), "Signal_Score(%)":round(percent_return,2), "signals_df":sig_df.reset_index(drop=True)}

FUND_KEYS=[("trailingPE","PE_TTM"),("forwardPE","PE_FWD"),("priceToBook","P_B"),("enterpriseToEbitda","EV_EBITDA"),("returnOnEquity","ROE"),("returnOnAssets","ROA"),("debtToEquity","D_E"),("dividendYield","DIV_YIELD"),("revenueGrowth","REV_GROWTH"),("earningsGrowth","EPS_GROWTH")]
def fetch_fundamentals_yf(ticker):
    p=cache_path(ticker,"fund")
    if cache_is_fresh_days(p,FUND_TTL_DAYS):
        try: df=pd.read_csv(p); return dict(zip(df["key"], df["value"]))
        except Exception: pass
    out={}
    try:
        info=yf.Ticker(ticker).info
        for k,alias in FUND_KEYS:
            v=info.get(k,np.nan); out[alias]=np.nan if v is None else v
        pd.DataFrame({"key":list(out.keys()),"value":list(out.values())}).to_csv(p,index=False)
        time.sleep(0.2+random.random()*0.3)
    except Exception:
        out={alias:np.nan for _,alias in FUND_KEYS}
    return out

def compute_feat_frame(df, idx_df=None, fund=None, fund_fallback=None):
    d=df.copy(); d["VOL"]=d.get("VOL",0).fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d["MFI"]=MFIIndicator(high=d["HIGH"],low=d["LOW"],close=d["CLOSE"],volume=d["VOL"],window=14).money_flow_index()
        d["RSI"]=RSIIndicator(close=d["CLOSE"],window=14).rsi()
        macd=MACD(close=d["CLOSE"],window_slow=26,window_fast=12,window_sign=9)
        d["MACD"]=macd.macd(); d["MACD_signal"]=macd.macd_signal(); d["MACD_hist"]=macd.macd_diff()
        d["SMA10"]=d["CLOSE"].rolling(10,min_periods=5).mean(); d["SMA30"]=d["CLOSE"].rolling(30,min_periods=10).mean()
        d=add_volatility_features(d)
        d["RET_1"]=d["CLOSE"].pct_change(1); d["RET_5"]=d["CLOSE"].pct_change(5); d["RET_10"]=d["CLOSE"].pct_change(10)
        d["DIST_SMA10"]=d["CLOSE"]/d["SMA10"]-1.0; d["DIST_SMA30"]=d["CLOSE"]/d["SMA30"]-1.0
    if idx_df is not None and not idx_df.empty:
        ii=idx_df.copy(); ii["VOL"]=ii.get("VOL",0).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ii["MFI"]=MFIIndicator(high=ii["HIGH"],low=ii["LOW"],close=ii["CLOSE"],volume=ii["VOL"],window=14).money_flow_index()
            ii["RSI"]=RSIIndicator(close=ii["CLOSE"],window=14).rsi()
        ii=add_volatility_features(ii); ii["RET_1"]=ii["CLOSE"].pct_change(1)
        aligned=d.join(ii[["CLOSE","MFI","RSI","RET_1"]].rename(columns={"CLOSE":"CLOSE_IDX","MFI":"MFI_IDX","RSI":"RSI_IDX","RET_1":"RET_1_IDX"}), how="left")
        for c in ["CLOSE_IDX","MFI_IDX","RSI_IDX","RET_1_IDX"]:
            if c in aligned.columns: aligned[c]=aligned[c].ffill()
        aligned["REL_RET_1"]=aligned["RET_1"]-aligned["RET_1_IDX"]; aligned["REL_RSI"]=aligned["RSI"]-aligned["RSI_IDX"]; aligned["REL_MFI"]=aligned["MFI"]-aligned["MFI_IDX"]
        aligned["CORR20"]=aligned["RET_1"].rolling(20,min_periods=10).corr(aligned["RET_1_IDX"])
        cov60=aligned["RET_1"].rolling(60,min_periods=30).cov(aligned["RET_1_IDX"]); var60=aligned["RET_1_IDX"].rolling(60,min_periods=30).var()
        beta60=cov60/var60; beta60[(var60.abs()<1e-18)]=np.nan; aligned["BETA60"]=beta60; d=aligned
    if fund:
        for k,v in fund.items():
            try: d[k]=float(v) if (v is not None and not pd.isna(v)) else np.nan
            except Exception: d[k]=np.nan
    if fund_fallback:
        for k in fund_fallback.keys():
            if k in d.columns: d[k]=d[k].astype(float).fillna(fund_fallback[k])
    d=replace_inf_with_nan(d); d=clip_extremes(d)
    must=["CLOSE","RET_1","MFI","RSI","SMA10","SMA30","CLOSE_IDX","RET_1_IDX","MFI_IDX","RSI_IDX","CORR20","BETA60","ATR14","ATR_ratio","Volatility14","BBWidth20","HV20"]
    d=d.dropna(subset=[c for c in must if c in d.columns])
    return d

def make_labels(d, horizon=FORWARD_HORIZON, thr=TARGET_THRESHOLD):
    d=d.copy(); d["FWD_RET"]=d["CLOSE"].shift(-horizon)/d["CLOSE"]-1.0; d["y"]=(d["FWD_RET"]>thr).astype(int)
    d=d.dropna(subset=["FWD_RET","y"]); return d

def fetch_wig20_frame(start_dt, end_dt):
    df_idx=fetch_from_stooq(WIG20_SYM); return prepare_stooq_df(df_idx, start_dt, end_dt)

def save_plots_pdf(overall_true, overall_proba, last_true=None, last_proba=None, out_pdf_path="model_plots.pdf"):
    with PdfPages(out_pdf_path) as pdf:
        fpr,tpr,_=roc_curve(overall_true, overall_proba); fig=plt.figure(); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (overall)"); pdf.savefig(fig); plt.close(fig)
        prec,rec,_=precision_recall_curve(overall_true, overall_proba); fig=plt.figure(); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (overall)"); pdf.savefig(fig); plt.close(fig)

def sanitize_matrix(X, fallback=None):
    X=X.replace([np.inf,-np.inf],np.nan)
    if fallback is None: fallback=X.median(numeric_only=True)
    X=X.fillna(fallback); X=X.clip(lower=-1e6, upper=1e6); return X, fallback

def walk_forward_evaluate(feat_frames, train_days, test_days, step_days, use_resampler=IMBLEARN_OK):
    all_feat=pd.concat(feat_frames,axis=0,sort=False).sort_index()
    X_cols=[c for c in all_feat.columns if c not in ("y","FWD_RET","CLOSE","Ticker")]
    start_time=all_feat.index.min(); end_time=all_feat.index.max()
    t0_list=[]; t0=start_time+dt.timedelta(days=train_days)
    while True:
        train_start=t0-dt.timedelta(days=train_days); train_end=t0; test_end=train_end+dt.timedelta(days=test_days)
        if test_end>end_time: break
        t0_list.append(t0); t0=t0+dt.timedelta(days=step_days)
    window_metrics=[]; overall_true=[]; overall_proba=[]; last_model=None
    for t0 in tqdm(t0_list, desc="AI: walk-forward", mininterval=0.2):
        train_start=t0-dt.timedelta(days=train_days); train_end=t0; test_end=train_end+dt.timedelta(days=test_days)
        train=all_feat.loc[(all_feat.index>train_start)&(all_feat.index<=train_end)]
        test =all_feat.loc[(all_feat.index>train_end)&(all_feat.index<=test_end)]
        if len(train)<100 or len(test)<15: continue
        X_train,y_train=train[X_cols].copy(),train["y"].copy(); X_test,y_test=test[X_cols].copy(),test["y"].copy()
        X_train,med=sanitize_matrix(X_train,None); X_test,_=sanitize_matrix(X_test,med)
        if use_resampler:
            try: X_train,y_train=SMOTEENN(random_state=RANDOM_STATE).fit_resample(X_train,y_train)
            except Exception: pass
        clf=RandomForestClassifier(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,random_state=RANDOM_STATE,class_weight=None if use_resampler else CLASS_WEIGHT,n_jobs=-1)
        clf.fit(X_train,y_train)
        proba=clf.predict_proba(X_test)[:,1]; y_true=y_test.values
        auc=roc_auc_score(y_true,proba) if len(np.unique(y_true))>1 else np.nan
        ap =average_precision_score(y_true,proba) if len(y_true)>0 else np.nan
        y_pred=(proba>=0.5).astype(int); acc=accuracy_score(y_true,y_pred) if len(y_true)>0 else np.nan
        prec=precision_score(y_true,y_pred,zero_division=0); rec=recall_score(y_true,y_pred,zero_division=0)
        window_metrics.append({"Train_Start":train_start.date(),"Train_End":train_end.date(),"Test_End":test_end.date(),"N_train":len(X_train),"N_test":len(X_test),"AUC":auc,"AP":ap,"ACC":acc,"Precision":prec,"Recall":rec})
        overall_true.append(y_true); overall_proba.append(proba); last_model=clf
    if not window_metrics: return None, None, None, None, X_cols
    metrics_df=pd.DataFrame(window_metrics); overall_true=np.concatenate(overall_true); overall_proba=np.concatenate(overall_proba)
    return metrics_df, overall_true, overall_proba, last_model, X_cols

def predict_today_with_model(summary_df, model, X_cols, today, idx_df, fund_table, fund_fallback, train_days=AI_TRAIN_DAYS):
    end_dt=today; start_dt=today-dt.timedelta(days=train_days); rows=[]
    for _,row in summary_df.iterrows():
        ticker=str(row["Ticker"]).strip()
        stq=fetch_from_stooq(ticker); dfr=prepare_stooq_df(stq,start_dt,end_dt) if stq is not None else None
        if dfr is None or dfr.empty: continue
        fund=fund_table.get(ticker,{})
        d=compute_feat_frame(dfr, idx_df=idx_df, fund=fund, fund_fallback=fund_fallback)
        if d.empty: continue
        x=d.iloc[[-1]].copy(); x=replace_inf_with_nan(x); x=clip_extremes(x); med=x.median(numeric_only=True); x=x.fillna(med).replace([np.inf,-np.inf],np.nan).fillna(0.0)
        missing=[c for c in X_cols if c not in x.columns]; if missing: continue
        proba=float(model.predict_proba(x[X_cols])[:,1][0])
        rows.append({"Ticker":ticker,"AI_Proba":proba,"Date":d.index[-1].date()}); time.sleep(0.005+random.random()*0.01)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Ticker","AI_Proba","Date"])

def build_daily_signals_with_ai(summary_df, today, pred_df):
    end_dt=today; start_dt=today-dt.timedelta(days=LOOKBACK_DAYS); daily_rows=[]
    for _,row in summary_df.iterrows():
        ticker=str(row["Ticker"]).strip(); mfi_lower=float(row["mfi_lower"]); mfi_upper=float(row["mfi_upper"]); rsi_lower=float(row["rsi_lower"]); rsi_upper=float(row["rsi_upper"])
        stq=fetch_from_stooq(ticker); dfr=prepare_stooq_df(stq,start_dt,end_dt) if stq is not None else None
        if dfr is None or dfr.empty: continue
        d=dfr.copy(); d["VOL"]=d.get("VOL",0).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d["MFI"]=MFIIndicator(high=d["HIGH"],low=d["LOW"],close=d["CLOSE"],volume=d["VOL"],window=14).money_flow_index()
            d["RSI"]=RSIIndicator(close=d["CLOSE"],window=14).rsi()
            macd=MACD(close=d["CLOSE"],window_slow=26,window_fast=12,window_sign=9); d["MACD"]=macd.macd(); d["MACD_signal"]=macd.macd_signal()
            d["SMA10"]=d["CLOSE"].rolling(10,min_periods=5).mean(); d["SMA30"]=d["CLOSE"].rolling(30,min_periods=10).mean()
        d=d.dropna(subset=["MFI","RSI","MACD","MACD_signal","SMA10","SMA30"]); if d.empty: continue
        latest=d.iloc[-1]; macd_trend="wzrostowy" if latest["MACD"]>=latest["MACD_signal"] else "spadkowy"; sma_trend="wzrostowy" if latest["SMA10"]>=latest["SMA30"] else "spadkowy"
        if (latest["MFI"]<mfi_lower) and (latest["RSI"]<rsi_lower): signal="buy"
        elif (latest["MFI"]>mfi_upper) or (latest["RSI"]>rsi_upper): signal="sell"
        else: signal="hold"
        daily_rows.append({"Ticker":ticker,"Date":d.index[-1].date(),"Close":float(latest["CLOSE"]),"MFI":float(latest["MFI"]),"RSI":float(latest["RSI"]),"MACD_Trend":macd_trend,"SMA_Trend":sma_trend,"Signal":signal})
    daily_df=pd.DataFrame(daily_rows); merged=daily_df.merge(pred_df[["Ticker","AI_Proba"]],on="Ticker",how="left")
    def ai_filter(r):
        if r["Signal"]=="buy" and r.get("AI_Proba",np.nan)>=AI_PROBA_THRESHOLD: return "buy"
        if r["Signal"]=="sell": return "sell"
        return "hold"
    merged["AI_Filtered_Signal"]=merged.apply(ai_filter,axis=1); return merged

def save_text_summary_pdf(metrics_df, topN_df, topN_filtered_df, today, out_pdf_path):
    lines=[]
    if metrics_df is not None and not metrics_df.empty:
        med_auc=metrics_df["AUC"].median(skipna=True); med_ap=metrics_df["AP"].median(skipna=True); mean_acc=metrics_df["ACC"].mean(skipna=True); wins=len(metrics_df)
        lines+=["Raport dzienny AI (skrót)", f"Data: {today.strftime('%Y-%m-%d')}", f"Okna walk-forward: {wins}", f"Mediana AUC: {med_auc:.3f}" if pd.notnull(med_auc) else "Mediana AUC: n/a", f"Mediana AP:  {med_ap:.3f}" if pd.notnull(med_ap) else "Mediana AP: n/a", f"Średnie ACC: {mean_acc:.3f}" if pd.notnull(mean_acc) else "Średnie ACC: n/a", ""]
    else:
        lines+=["Raport dzienny AI (skrót)", f"Data: {today.strftime('%Y-%m-%d')}", "Brak metryk walk-forward.", ""]
    lines.append("Top sygnały wg AI_Proba:")
    if topN_df is not None and not topN_df.empty:
        for i,r in topN_df.iterrows(): lines.append(f"{i+1}. {r['Ticker']}: AI_Proba={r['AI_Proba']:.3f}, Signal={r.get('Signal','n/a')}")
    else: lines.append("(brak)")
    lines.append(""); lines.append("Top sygnały (AI_Filtered_Signal == 'buy'):")
    if topN_filtered_df is not None and not topN_filtered_df.empty:
        for i,r in topN_filtered_df.iterrows(): lines.append(f"{i+1}. {r['Ticker']}: AI_Proba={r['AI_Proba']:.3f}")
    else: lines.append("(brak)")
    with PdfPages(out_pdf_path) as pdf:
        fig=plt.figure(figsize=(8.27,11.69)); plt.axis("off"); plt.text(0.05,0.95,"\n".join(lines),fontsize=10,va="top",family="monospace"); pdf.savefig(fig); plt.close(fig)

def main():
    today=dt.datetime.now(); end_dt=today; start_dt=today-dt.timedelta(days=LOOKBACK_DAYS)
    tickers=fetch_bankier_tickers(); print("Pobrane tickery (%d):"%len(tickers), tickers)
    if not tickers: print("Brak tickerów – koniec."); return
    results=[]; signals=[]
    for t in tqdm(tickers, desc="Backtest MFI/RSI (Stooq)"):
        stq=fetch_from_stooq(t); df_raw=prepare_stooq_df(stq,start_dt,end_dt) if stq is not None else None
        out=process_ticker(t, df_raw); 
        if out is None: continue
        results.append({k:v for k,v in out.items() if k!="signals_df"}); signals.append(out["signals_df"]); time.sleep(0.04+random.random()*0.06)
    if not results: print("Brak wyników – za mało danych."); return
    summary_df=pd.DataFrame(results).sort_values("Signal_Score(%)",ascending=False).reset_index(drop=True)
    signals_df=pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=["Date","Signal","Ticker"])
    out_name=f"wyniki_{today.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(out_name, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Podsumowanie", index=False)
        signals_df.to_excel(w, sheet_name="Sygnały", index=False)
    print("OK. Zapisano:", out_name)
    ai_end=today; ai_start=today-dt.timedelta(days=AI_TRAIN_DAYS+AI_TEST_DAYS+200)
    idx_df=fetch_wig20_frame(ai_start, ai_end)
    fundamentals_table={}; fund_rows=[]
    for t in tqdm(summary_df["Ticker"].tolist(), desc="Pobieranie fundamentów (Yahoo)"):
        fund=fetch_fundamentals_yf(t); fundamentals_table[t]=fund; row={"Ticker":t}; row.update(fund); fund_rows.append(row)
    fund_df_all=pd.DataFrame(fund_rows)
    fund_fallback=(fund_df_all.drop(columns=["Ticker"],errors="ignore").median(numeric_only=True).to_dict())
    feat_frames=[]
    for t in tqdm(summary_df["Ticker"].tolist(), desc="AI: cechy + etykiety (z fundamentami)"):
        stq=fetch_from_stooq(str(t)); dfr=prepare_stooq_df(stq, ai_start, ai_end) if stq is not None else None
        if dfr is None or dfr.empty: continue
        d=compute_feat_frame(dfr, idx_df=idx_df, fund=fundamentals_table.get(t,{}), fund_fallback=fund_fallback); d=make_labels(d)
        if d.empty: continue
        d["Ticker"]=str(t); feat_frames.append(d); time.sleep(0.02+random.random()*0.03)
    if not feat_frames: print("AI: brak danych do walk-forward."); return out_name, summary_df, signals_df
    metrics_df, overall_true, overall_proba, last_model, X_cols = walk_forward_evaluate(feat_frames, AI_TRAIN_DAYS, AI_TEST_DAYS, AI_STEP_DAYS, use_resampler=IMBLEARN_OK)
    if metrics_df is None:
        print("AI: WF nie wyszedł — fallback 80/20.")
        all_feat=pd.concat(feat_frames,axis=0,sort=False).sort_index()
        X_cols=[c for c in all_feat.columns if c not in ("y","FWD_RET","CLOSE","Ticker")]
        cutoff=int(len(all_feat)*0.8); train=all_feat.iloc[:cutoff].copy(); valid=all_feat.iloc[cutoff:].copy()
        X_train,med=sanitize_matrix(train[X_cols],None); valid_X,_=sanitize_matrix(valid[X_cols],med)
        if IMBLEARN_OK:
            try: X_train,y_train=SMOTEENN(random_state=RANDOM_STATE).fit_resample(X_train,train["y"])
            except Exception: y_train=train["y"]
        else: y_train=train["y"]
        clf=RandomForestClassifier(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,random_state=RANDOM_STATE,class_weight=None if IMBLEARN_OK else CLASS_WEIGHT,n_jobs=-1); clf.fit(X_train,y_train); last_model=clf
        try:
            proba_valid=clf.predict_proba(valid_X)[:,1]; overall_true=valid["y"].values; overall_proba=proba_valid
            metrics_df=pd.DataFrame([{"Train_Start":train.index.min().date(),"Train_End":train.index.max().date(),"Test_End":valid.index.max().date(),"N_train":len(X_train),"N_test":len(valid),"AUC":roc_auc_score(valid["y"],proba_valid) if len(valid)>0 else np.nan,"AP":average_precision_score(valid["y"],proba_valid) if len(valid)>0 else np.nan,"ACC":accuracy_score(valid["y"],(proba_valid>=0.5).astype(int)) if len(valid)>0 else np.nan,"Precision":precision_score(valid["y"],(proba_valid>=0.5).astype(int),zero_division=0) if len(valid)>0 else 0,"Recall":recall_score(valid["y"],(proba_valid>=0.5).astype(int),zero_division=0) if len(valid)>0 else 0}])
        except Exception:
            overall_true,overall_proba=np.array([]),np.array([])
    stats_name=f"model_stats_{today.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(stats_name, engine="openpyxl") as w:
        metrics_df.to_excel(w, sheet_name="Model_Stats", index=False)
        try:
            fi=pd.DataFrame({"Feature":X_cols,"Importance":last_model.feature_importances_}).sort_values("Importance",ascending=False).reset_index(drop=True)
            fi.to_excel(w, sheet_name="Feature_Importance", index=False)
        except Exception: pass
        pd.DataFrame(fund_rows).to_excel(w, sheet_name="Fundamentals", index=False)
    print("OK. Zapisano metryki i fundamenty:", stats_name)
    plots_name=f"model_plots_{today.strftime('%Y%m%d')}.pdf"
    if overall_true is not None and len(overall_true)>0: save_plots_pdf(overall_true, overall_proba, None, None, out_pdf_path=plots_name)
    else:
        with PdfPages(plots_name) as pdf: fig=plt.figure(); plt.axis("off"); plt.text(0.1,0.9,"Brak danych do ROC/PR",fontsize=12,va="top"); pdf.savefig(fig); plt.close(fig)
    print("OK. Zapisano wykresy:", plots_name)
    idx_today=fetch_wig20_frame(today-dt.timedelta(days=AI_TRAIN_DAYS), today)
    pred_df=predict_today_with_model(summary_df, last_model, X_cols, today, idx_today, fundamentals_table, fund_fallback, train_days=AI_TRAIN_DAYS)
    merged=build_daily_signals_with_ai(summary_df, today, pred_df)
    topN_df=merged.dropna(subset=["AI_Proba"]).sort_values("AI_Proba",ascending=False).head(TOP_N).reset_index(drop=True)
    topN_filtered_df=merged.query("AI_Filtered_Signal == 'buy'").dropna(subset=["AI_Proba"]).sort_values("AI_Proba",ascending=False).head(TOP_N).reset_index(drop=True)
    out_ai=f"sygnaly_{today.strftime('%Y%m%d')}_ai.xlsx"
    with pd.ExcelWriter(out_ai, engine="openpyxl") as w:
        merged.to_excel(w, sheet_name="Daily_Signals_AI", index=False)
        topN_df.to_excel(w, sheet_name=f"Top{TOP_N}_By_AI_Proba", index=False)
        topN_filtered_df.to_excel(w, sheet_name=f"Top{TOP_N}_Filtered_Buys", index=False)
    print("OK. Zapisano dzienne sygnały AI + TOP-N:", out_ai)
    summary_pdf=f"model_summary_{today.strftime('%Y%m%d')}.pdf"; 
    save_text_summary_pdf(metrics_df, topN_df, topN_filtered_df, today, summary_pdf); print("OK. Zapisano raport PDF:", summary_pdf)
    return out_name, summary_df, signals_df, stats_name, plots_name, out_ai, summary_pdf

if __name__ == "__main__":
    main()
