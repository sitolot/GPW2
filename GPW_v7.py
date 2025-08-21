# -*- coding: utf-8 -*-
"""
GPW WIG140 Backtest + AI — FULL (walk-forward, BEZ FUNDAMENTÓW)
- Ceny: Stooq (spółki + WIG20)
- Strategia bazowa: MFI/RSI (grid)
- Cechy AI: techniczne, zmienność/ATR/HV, relacje do WIG20 (żadnych fundamentów)
- Balans klas: SMOTEENN (jeśli imblearn dostępny), inaczej class_weight="balanced"
- Model: RandomForest (n_jobs=-1)
- Raporty: wyniki_*.xlsx, model_stats_*.xlsx, model_plots_*.pdf, model_summary_*.pdf, sygnaly_*_ai.xlsx
"""
import sys
import os, time, random, itertools, warnings, datetime as dt
from io import StringIO
import numpy as np, pandas as pd, requests
import smtplib
from email.header import Header
from email.utils import formataddr
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pandas.tseries.offsets import BDay
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm import tqdm
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, AccDistIndexIndicator, EaseOfMovementIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import DonchianChannel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# ==== PERSISTENCE / CLI ====
import argparse, json
from pathlib import Path
from joblib import dump, load

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"; OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = OUT_DIR / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUT_DIR / "runs"; RUN_DIR.mkdir(parents=True, exist_ok=True)

# ===== opcjonalny imblearn =====
IMBLEARN_OK = True
try:
    from imblearn.combine import SMOTEENN
except Exception:
    IMBLEARN_OK = False

# ===== opcjonalny numba =====
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

# === ENSEMBLE / KALIBRACJA / RANK-BAGGING ===
CALIBRATION_METHOD = "isotonic"   # "isotonic" lub "sigmoid"
CALIBRATION_FRAC   = 0.15         # ile końcówki TRAIN wykorzystać na kalibrację
VOTING_WEIGHTS     = {"rf":0.25, "xgb":0.35, "lgb":0.40}  # startowe; będą aktualizowane z AP
USE_RANK_ENSEMBLE  = True         # rank-średnia obok zwykłej średniej proba
BAG_LAST_K         = 3            # ile OSTATNICH okien WF brać do baggingu podczas predict

# === OPCJONALNY BASE MODEL SEKWENCYJNY (LSTM/TCN) ===
USE_SEQ_BASE       = False        # włączysz, gdy będziesz gotów
SEQ_LEN            = 30           # ile dni w sekwencji do NN

TOP_N = 10

CACHE_DIR = "cache_gpwwig"; os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_DAYS = 1

MFI_LOWER_RANGE = [15, 20, 25, 30, 35, 40]
MFI_UPPER_RANGE = [65, 70, 75, 80]
RSI_LOWER_RANGE = [20, 30, 40, 50]
RSI_UPPER_RANGE = [60, 70, 80]

STOOQ_URLS = [
    "https://stooq.com/q/d/l/?s={sym}&i=d",
    "https://stooq.pl/q/d/l/?s={sym}&i=d"
]
WIG20_SYM = "wig20"

# ===================== session/cache =====================
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

# ===================== tickery =====================
def fetch_bankier_tickers():
    """Szuka tickrów WIG140 na Bankier (prosty parser tabeli)."""
    url = "https://www.bankier.pl/inwestowanie/profile/quote.html?symbol=WIG140"
    try:
        s = _requests_session(); r = s.get(url, timeout=10); r.raise_for_status()
    except Exception as e:
        print("Błąd pobierania listy:", e); return []
    soup = BeautifulSoup(r.text, "html.parser")
    candidate = None
    for table in soup.find_all("table"):
        txt = table.get_text(" ", strip=True).lower()
        if any(k in txt for k in ("ticker","symbol","skrót","nazwa")):
            candidate = table; break
    if candidate is None: return []
    tks = []
    for tr in candidate.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds)>=2:
            tk = tds[1].get_text(strip=True)
            if tk:
                tks.append(tk if tk.endswith(".WA") else f"{tk}.WA")
    return list(dict.fromkeys(tks))

# ===================== Stooq utils =====================
STQ_PL2EN = {
    "Data":"Date","Otwarcie":"Open","Najwyzszy":"High","Najwyższy":"High",
    "Najnizszy":"Low","Najniższy":"Low","Zamkniecie":"Close","Zamknięcie":"Close","Wolumen":"Volume"
}
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
            df=pd.read_csv(StringIO(r.text))
            if df is None or df.empty: continue
            df=normalize_stooq_headers(df); df.to_csv(p,index=False); return df
        except Exception:
            continue
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

# ===================== Auto prefilter płynności/ruchu =====================
LIQ_LOOKBACK_DAYS = 180
PREFILTER_MODE = os.getenv("PREFILTER_MODE", "balanced").lower()  # 'strict' | 'balanced' | 'loose'

def compute_liquidity_metrics(df):
    if df is None or df.empty:
        return None
    df = df.tail(LIQ_LOOKBACK_DAYS).copy()
    if df.empty:
        return None
    m = {}
    m['active_ratio'] = float((df['VOL'] > 0).mean())
    m['avg_price'] = float(df['CLOSE'].mean())
    m['avg_vol'] = float(df['VOL'].mean())
    m['avg_turnover_pln'] = float((df['VOL'] * df['CLOSE']).mean())
    spread_proxy = (df['HIGH'] - df['LOW']) / df['CLOSE'].replace(0, np.nan)
    m['avg_spread_proxy'] = float(np.nanmean(spread_proxy.values))
    tr1 = (df['HIGH'] - df['LOW']).abs()
    tr2 = (df['HIGH'] - df['CLOSE'].shift(1)).abs()
    tr3 = (df['LOW']  - df['CLOSE'].shift(1)).abs()
    TR  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    m['ATR14'] = float(TR.rolling(14).mean().iloc[-1])
    prev_close = df['CLOSE'].shift(1)
    gap = (df['CLOSE'] - prev_close).abs() / prev_close.replace(0, np.nan)
    m['gap_ratio_2pct'] = float((gap > 0.02).mean())
    return m

def prefilter_collect_metrics(tickers, start_dt, end_dt, fetch_fn, prep_fn):
    rows = []
    for t in tqdm(tickers, desc="Pre-filter: zbieranie metryk"):
        stq = fetch_fn(t)
        df_raw = prep_fn(stq, start_dt, end_dt) if stq is not None else None
        m = compute_liquidity_metrics(df_raw)
        r = {"Ticker": t}
        if m is not None:
            r.update(m)
        rows.append(r)
    return pd.DataFrame(rows)

def _q(series, q, default=np.nan):
    try:
        return float(np.nanpercentile(series.values.astype(float), q*100))
    except Exception:
        return default

def auto_thresholds(metrics_df: pd.DataFrame, mode: str = "balanced") -> dict:
    mode = mode.lower()
    if mode == "strict":
        q_min, q_max = 0.40, 0.80
    elif mode == "loose":
        q_min, q_max = 0.20, 0.90
    else:
        q_min, q_max = 0.30, 0.85

    df = metrics_df.copy()
    th = {}
    th['min_price']        = max(1.5, _q(df['avg_price'],        q_min, default=1.5))
    th['min_avg_vol']      = _q(df['avg_vol'],                   q_min, default=0.0)
    th['min_turnover_pln'] = _q(df['avg_turnover_pln'],          q_min, default=0.0)
    th['min_active_ratio'] = max(0.60, _q(df['active_ratio'],    q_min, default=0.60))
    th['min_ATR14']        = _q(df['ATR14'],                     q_min, default=np.nan)
    th['max_spread_proxy'] = min(0.12, _q(df['avg_spread_proxy'], q_max, default=0.12))
    th['max_gap_ratio']    = _q(df['gap_ratio_2pct'],             q_max, default=np.nan)
    return th

def row_passes(row: pd.Series, th: dict) -> bool:
    if not pd.isna(th.get('min_price', np.nan))        and row.get('avg_price',        0) < th['min_price']       : return False
    if not pd.isna(th.get('min_avg_vol', np.nan))      and row.get('avg_vol',          0) < th['min_avg_vol']     : return False
    if not pd.isna(th.get('min_turnover_pln', np.nan)) and row.get('avg_turnover_pln', 0) < th['min_turnover_pln']: return False
    if not pd.isna(th.get('min_active_ratio', np.nan)) and row.get('active_ratio',     0) < th['min_active_ratio']: return False
    if not pd.isna(th.get('min_ATR14', np.nan)):
        atr = row.get('ATR14', np.nan)
        if not pd.isna(atr) and atr < th['min_ATR14']: return False
    sp = row.get('avg_spread_proxy', np.nan)
    if not pd.isna(th.get('max_spread_proxy', np.nan)) and not pd.isna(sp) and sp > th['max_spread_proxy']: return False
    gr = row.get('gap_ratio_2pct', np.nan)
    if not pd.isna(th.get('max_gap_ratio', np.nan)) and not pd.isna(gr) and gr > th['max_gap_ratio']: return False
    return True

def prefilter_tickers_auto(tickers, start_dt, end_dt, fetch_fn, prep_fn, mode=PREFILTER_MODE):
    metrics_df = prefilter_collect_metrics(tickers, start_dt, end_dt, fetch_fn, prep_fn)
    if metrics_df.empty:
        return tickers, metrics_df, {}
    th = auto_thresholds(metrics_df, mode=mode)
    kept = [r['Ticker'] for _, r in metrics_df.iterrows() if row_passes(r, th)]
    print(f"[Prefilter:{mode}] Przeszło {len(kept)}/{len(tickers)} spółek.")
    print("Progi (auto):", th)
    return kept, metrics_df, th

# ===================== narzędzia danych/cech =====================
def replace_inf_with_nan(df):
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c]=df[c].replace([np.inf,-np.inf],np.nan)
    return df

def clip_extremes(df):
    # tylko techniczne
    bounds = {
        "ATR_ratio": (0, 1.5),
        "BBWidth20": (0, 3),
        "HV20": (0, 3),
        "Volatility14": (0, 1),

        # nowo dodane:
        "PercentB20": (0, 1.2),
        "DONCH_POS": (0, 1.2),
        "CLV": (-1.2, 1.2),
        "RangePct": (0, 0.30),
        "OBV_Z20": (-5, 5),
        "EOM_Z20": (-5, 5),
        "DownVol20": (0, 3),
        "IdioHV20": (0, 3),
        "Skew60": (-3, 3),
        "Kurt60": (-1, 10)
    }
    df = df.copy()
    for col,(lo,hi) in bounds.items():
        if col in df.columns: df[col]=df[col].clip(lower=lo,upper=hi)
    return df

def add_volatility_features(d):
    d=d.copy(); prev_close=d["CLOSE"].shift(1)
    tr = pd.concat([
        d["HIGH"]-d["LOW"],
        (d["HIGH"]-prev_close).abs(),
        (d["LOW"]-prev_close).abs()
    ],axis=1).max(axis=1)
    d["ATR14"]=tr.rolling(14,min_periods=7).mean()
    d["ATR_ratio"]=d["ATR14"]/d["CLOSE"]
    d["RET_D"]=d["CLOSE"].pct_change()
    d["Volatility14"]=d["RET_D"].rolling(14,min_periods=7).std()
    sma20=d["CLOSE"].rolling(20,min_periods=10).mean()
    std20=d["CLOSE"].rolling(20,min_periods=10).std()
    bbwidth=(4*std20)/sma20
    bbwidth[(sma20.abs()<1e-12)]=np.nan
    d["BBWidth20"]=bbwidth
    logret=np.log(d["CLOSE"]).diff()
    d["HV20"]=logret.rolling(20,min_periods=10).std()*np.sqrt(252.0)
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
    df=add_indicators(df_raw)
    if df.empty: return None
    close_values=df["CLOSE"].to_numpy(np.float64)
    mfi_values=df["MFI"].to_numpy(np.float64)
    rsi_values=df["RSI"].to_numpy(np.float64)
    best_cap=-np.inf; best_params=None; best_signals=None; best_index=None
    for mfi_low,mfi_up,rsi_low,rsi_up in itertools.product(MFI_LOWER_RANGE,MFI_UPPER_RANGE,RSI_LOWER_RANGE,RSI_UPPER_RANGE):
        if mfi_low>=mfi_up or rsi_low>=rsi_up: continue
        cap, sig = backtest_arrays(close_values,mfi_values,rsi_values,mfi_low,mfi_up,rsi_low,rsi_up,INITIAL_CAPITAL)
        if cap>best_cap:
            best_cap=cap
            best_params={"mfi_lower":mfi_low,"mfi_upper":mfi_up,"rsi_lower":rsi_low,"rsi_upper":rsi_up}
            best_signals=sig.copy(); best_index=df.index
    if best_params is None: return None
    sig_df=pd.DataFrame({"Date":best_index[1:], "Signal":best_signals[1:].astype(np.int8)})
    sig_df=sig_df[sig_df["Signal"]!=0].copy(); sig_df["Ticker"]=ticker
    percent_return=(best_cap/INITIAL_CAPITAL-1.0)*100.0
    return {"Ticker":ticker, **best_params, "Final_Capital":round(best_cap,2),
            "Signal_Score(%)":round(percent_return,2), "signals_df":sig_df.reset_index(drop=True)}

def compute_feat_frame(df, idx_df=None):
    d=df.copy(); d["VOL"]=d.get("VOL",0).fillna(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d["MFI"]=MFIIndicator(high=d["HIGH"],low=d["LOW"],close=d["CLOSE"],volume=d["VOL"],window=14).money_flow_index()
        d["RSI"]=RSIIndicator(close=d["CLOSE"],window=14).rsi()
        macd=MACD(close=d["CLOSE"],window_slow=26,window_fast=12,window_sign=9)
        d["MACD"]=macd.macd(); d["MACD_signal"]=macd.macd_signal(); d["MACD_hist"]=macd.macd_diff()
        d["SMA10"]=d["CLOSE"].rolling(10,min_periods=5).mean()
        d["SMA30"]=d["CLOSE"].rolling(30,min_periods=10).mean()
        d=add_volatility_features(d)
        d["RET_1"]=d["CLOSE"].pct_change(1)
        d["RET_5"]=d["CLOSE"].pct_change(5)
        d["RET_10"]=d["CLOSE"].pct_change(10)
        d["DIST_SMA10"]=d["CLOSE"]/d["SMA10"]-1.0
        d["DIST_SMA30"]=d["CLOSE"]/d["SMA30"]-1.0
    if idx_df is not None and not idx_df.empty:
        ii=idx_df.copy(); ii["VOL"]=ii.get("VOL",0).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ii["MFI"]=MFIIndicator(high=ii["HIGH"],low=ii["LOW"],close=ii["CLOSE"],volume=ii["VOL"],window=14).money_flow_index()
            ii["RSI"]=RSIIndicator(close=ii["CLOSE"],window=14).rsi()
        ii=add_volatility_features(ii); ii["RET_1"]=ii["CLOSE"].pct_change(1)
        aligned=d.join(ii[["CLOSE","MFI","RSI","RET_1"]].rename(
            columns={"CLOSE":"CLOSE_IDX","MFI":"MFI_IDX","RSI":"RSI_IDX","RET_1":"RET_1_IDX"}), how="left")
        for c in ["CLOSE_IDX","MFI_IDX","RSI_IDX","RET_1_IDX"]:
            if c in aligned.columns: aligned[c]=aligned[c].ffill()
        aligned["REL_RET_1"]=aligned["RET_1"]-aligned["RET_1_IDX"]
        aligned["REL_RSI"]=aligned["RSI"]-aligned["RSI_IDX"]
        aligned["REL_MFI"]=aligned["MFI"]-aligned["MFI_IDX"]
        aligned["CORR20"]=aligned["RET_1"].rolling(20,min_periods=10).corr(aligned["RET_1_IDX"])
        cov60=aligned["RET_1"].rolling(60,min_periods=30).cov(aligned["RET_1_IDX"])
        var60=aligned["RET_1_IDX"].rolling(60,min_periods=30).var()
        beta60=cov60/var60; beta60[(var60.abs()<1e-18)]=np.nan
        aligned["BETA60"]=beta60; d=aligned
        # === NOWE CECHY (drop-in) ===
        # --- typowa cena & VWAP-y ---
        tp = (d["HIGH"] + d["LOW"] + d["CLOSE"]) / 3.0
        vol = d["VOL"].astype(float).fillna(0.0)

        v_num = (tp * vol).rolling(20, min_periods=10).sum()
        v_den = vol.rolling(20, min_periods=10).sum().replace(0, np.nan)
        d["VWAP20"] = v_num / v_den
        d["DIST_VWAP20"] = d["CLOSE"] / d["VWAP20"] - 1.0

        year_idx = d.index.year
        cum_num = (tp * vol).groupby(year_idx).cumsum()
        cum_den = vol.groupby(year_idx).cumsum().replace(0, np.nan)
        d["AVWAP_YTD"] = (cum_num / cum_den)
        d["DIST_AVWAP_YTD"] = d["CLOSE"] / d["AVWAP_YTD"] - 1.0

        # --- oscylatory / trend ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stoch = StochasticOscillator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], window=14, smooth_window=3)
            d["STOCH_K"] = stoch.stoch()
            d["STOCH_D"] = stoch.stoch_signal()
            d["WILLR"] = WilliamsRIndicator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], lbp=14).williams_r()
            d["CCI20"] = CCIIndicator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], window=20).cci()

            adx = ADXIndicator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], window=14)
            d["ADX14"] = adx.adx()
            d["DI_POS"] = adx.adx_pos()
            d["DI_NEG"] = adx.adx_neg()

        # --- Donchian 20: szerokość i pozycja w kanale ---
        don = DonchianChannel(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], window=20)
        hband = don.donchian_channel_hband()
        lband = don.donchian_channel_lband()
        width = (hband - lband).replace(0, np.nan)
        d["DONCH_WIDTH"] = width / d["CLOSE"]
        d["DONCH_POS"] = (d["CLOSE"] - lband) / width  # 0..1 (pozycja w kanale)

        # --- Bollinger %B (masz już BBWidth, ale %B dodaje informację o pozycji) ---
        sma20 = d["CLOSE"].rolling(20, min_periods=10).mean()
        std20 = d["CLOSE"].rolling(20, min_periods=10).std()
        upper = sma20 + 2.0 * std20
        lower = sma20 - 2.0 * std20
        denom = (upper - lower).replace(0, np.nan)
        d["PercentB20"] = (d["CLOSE"] - lower) / denom

        # --- wolumen / przepływy ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d["OBV"] = OnBalanceVolumeIndicator(close=d["CLOSE"], volume=vol).on_balance_volume()
            d["CMF20"] = ChaikinMoneyFlowIndicator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"], volume=vol,
                                                   window=20).chaikin_money_flow()
            d["ADL"] = AccDistIndexIndicator(high=d["HIGH"], low=d["LOW"], close=d["CLOSE"],
                                             volume=vol).acc_dist_index()
            eom = EaseOfMovementIndicator(high=d["HIGH"], low=d["LOW"], volume=vol, window=14).ease_of_movement()

            # z-score dla niereskalowanych wielkości:
            def zscore(s, w):
                m = s.rolling(w, min_periods=max(5, w // 3)).mean()
                sd = s.rolling(w, min_periods=max(5, w // 3)).std()
                return (s - m) / sd

            d["OBV_Z20"] = zscore(d["OBV"], 20)
            d["EOM_Z20"] = zscore(eom, 20)

        # --- zmienność / mikrostruktura ---
        rng = (d["HIGH"] - d["LOW"]).replace(0, np.nan)
        d["CLV"] = ((d["CLOSE"] - d["LOW"]) - (d["HIGH"] - d["CLOSE"])) / rng  # w [-1,1]
        d["RangePct"] = (d["HIGH"] - d["LOW"]) / d["CLOSE"]
        neg = d["RET_1"].clip(upper=0.0)
        d["DownVol20"] = neg.rolling(20, min_periods=10).std() * np.sqrt(252.0)
        if "REL_RET_1" in d.columns:
            d["IdioHV20"] = d["REL_RET_1"].rolling(20, min_periods=10).std() * np.sqrt(252.0)
        d["Skew60"] = d["RET_1"].rolling(60, min_periods=30).skew()
        d["Kurt60"] = d["RET_1"].rolling(60, min_periods=30).kurt()

        # --- SMA200 & dystans ---
        d["SMA200"] = d["CLOSE"].rolling(200, min_periods=100).mean()
        d["DIST_SMA200"] = d["CLOSE"] / d["SMA200"] - 1.0

        # --- momentum dłuższe horyzonty ---
        d["RET_20"] = d["CLOSE"].pct_change(20)
        d["RET_60"] = d["CLOSE"].pct_change(60)
        d["RET_120"] = d["CLOSE"].pct_change(120)

        # --- sezonowość zakodowana cyklicznie ---
        dow = d.index.dayofweek
        doy = d.index.dayofyear
        d["DOW_sin"] = np.sin(2 * np.pi * dow / 5.0)
        d["DOW_cos"] = np.cos(2 * np.pi * dow / 5.0)
        d["DOY_sin"] = np.sin(2 * np.pi * doy / 365.0)
        d["DOY_cos"] = np.cos(2 * np.pi * doy / 365.0)
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
        fpr,tpr,_=roc_curve(overall_true, overall_proba)
        fig=plt.figure(); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (overall)"); pdf.savefig(fig); plt.close(fig)
        prec,rec,_=precision_recall_curve(overall_true, overall_proba)
        fig=plt.figure(); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (overall)"); pdf.savefig(fig); plt.close(fig)

def sanitize_matrix(X, fallback=None):
    X=X.replace([np.inf,-np.inf],np.nan)
    if fallback is None: fallback=X.median(numeric_only=True)
    X=X.fillna(fallback); X=X.clip(lower=-1e6, upper=1e6); return X, fallback

def pick_threshold_for_precision(proba, y_true, min_precision=0.70):
    """Wybiera najniższy próg spełniający Precision >= min_precision,
    a wśród takich maksymalizuje Recall. Zwraca: (threshold, precision, recall)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)

    # Usuwamy NaN-y
    m = np.isfinite(prec) & np.isfinite(rec)
    prec, rec = prec[m], rec[m]
    # Uwaga: 'thr' ma o 1 element mniej niż prec/rec
    if len(thr) == len(prec) - 1:
        ok_mask = prec[:-1] >= min_precision
        candidates = np.where(ok_mask)[0]
        if len(candidates) == 0:
            # fallback: bierzemy punkt o największym Recall
            best_idx = int(np.nanargmax(rec[:-1]))
            return float(thr[best_idx]), float(prec[best_idx]), float(rec[best_idx])
        # Maksymalizuj Recall wśród kandydatów
        best_local = candidates[np.argmax(rec[:-1][candidates])]
        return float(thr[best_local]), float(prec[best_local]), float(rec[best_local])
    else:
        # Nietypowy przypadek długości; bezpieczny fallback
        # wybierz threshold odpowiadający max Recall przy Precision >= min_precision
        ok = prec >= min_precision
        if not np.any(ok):
            j = int(np.nanargmax(rec))
            return 0.50, float(prec[j]), float(rec[j])
        j = np.argmax(rec[ok])
        pos = np.where(ok)[0][j]
        
        return 0.50, float(prec[pos]), float(rec[pos])

def make_base_models(spw, random_state=RANDOM_STATE):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=random_state
    )
    xgb = XGBClassifier(
        n_estimators=1500, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.5,
        tree_method="hist", objective="binary:logistic",
        n_jobs=-1, random_state=random_state, scale_pos_weight=spw,
        eval_metric="aucpr"  
    )
    lgb = LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary", n_jobs=-1, random_state=random_state,
        scale_pos_weight=spw, verbose=-1
    )
    return rf, xgb, lgb

def build_oof_and_fit_bases(X_train, y_train, n_splits=3):
    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    spw = (neg / max(pos, 1))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --- (1) Stwórz bazowe modele
    rf, xgb, lgb = make_base_models(spw)

    # --- (2) Macierz OOF (NaN na start)
    oof_mat = np.full((len(X_train), 3), np.nan, dtype=float)

    # --- (3) K-fold (time-series) i wpisywanie predykcji bazowych do oof_mat
    for inner_tr_idx, inner_val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[inner_tr_idx], X_train.iloc[inner_val_idx]
        y_tr, y_val = y_train.iloc[inner_tr_idx], y_train.iloc[inner_val_idx]

        # Trenuj każdy model na wewnętrznym train
        rf.fit(X_tr, y_tr)

        # XGBoost: bez callbacks (kompatybilność), eval_metric może być w fit
        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # LightGBM: early_stopping przez callbacks
        import lightgbm as lgbm
        lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgbm.early_stopping(150, verbose=False), lgbm.log_evaluation(-1)]
        )

        # Predykcje na walidacji
        p_rf  = rf.predict_proba(X_val)[:, 1]
        p_xgb = xgb.predict_proba(X_val)[:, 1]
        p_lgb = lgb.predict_proba(X_val)[:, 1]

        # Wpisz do macierzy OOF (indeksami folda)
        oof_mat[inner_val_idx, 0] = p_rf
        oof_mat[inner_val_idx, 1] = p_xgb
        oof_mat[inner_val_idx, 2] = p_lgb

    # --- (4) IMPUTACJA OOF -> braków / inf
    # Średnie kolumnowe (gdy same NaN-y w kolumnie -> fallback do 0.5)
    col_means = np.nanmean(oof_mat, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.5)
    # Wstaw średnie w miejsca NaN
    nan_rows, nan_cols = np.where(np.isnan(oof_mat))
    if len(nan_rows) > 0:
        oof_mat[nan_rows, nan_cols] = col_means[nan_cols]
    # Utnij inf/NaN na wszelki wypadek
    oof_mat = np.nan_to_num(oof_mat, nan=0.5, posinf=1.0, neginf=0.0)

    # (opcjonalnie) szybka diagnostyka
    # print("OOF stats per col:", np.nanmin(oof_mat, axis=0), np.nanmax(oof_mat, axis=0))

    # --- (5) DataFrame OOF dla meta-klasera
    oof = pd.DataFrame(oof_mat, index=X_train.index, columns=["rf", "xgb", "lgb"])

    # --- (6) Meta-klaser
    meta = LogisticRegression(max_iter=10_000, C=1.0, solver="lbfgs")
    meta.fit(oof.values, y_train.values)

    # --- (7) Fit baz na CAŁYM X_train do późniejszych predykcji na X_test
    rf_full, xgb_full, lgb_full = make_base_models(spw)

    rf_full.fit(X_train, y_train)

    xgb_full.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )

    import lightgbm as lgbm
    lgb_full.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_metric="average_precision",
        callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(-1)]
    )

    base_bundle = {"rf": rf_full, "xgb": xgb_full, "lgb": lgb_full, "meta": meta}
    return oof, base_bundle

def _split_calibration_tail(X_train, y_train, frac=0.15):
    """Odrywa ogon train na kalibrację. Zwraca: X_fit, y_fit, X_cal, y_cal."""
    n = len(X_train)
    k = max(1, int(n * frac))
    X_fit = X_train.iloc[:-k].copy() if n > k else X_train.iloc[:0].copy()
    y_fit = y_train.iloc[:-k].copy() if n > k else y_train.iloc[:0].copy()
    X_cal = X_train.iloc[-k:].copy()
    y_cal = y_train.iloc[-k:].copy()
    if len(X_fit) < 50:  # awaryjnie gdy za mało
        X_fit, y_fit = X_train.copy(), y_train.copy()
        X_cal, y_cal = X_train.iloc[-k:].copy(), y_train.iloc[-k:].copy()
    return X_fit, y_fit, X_cal, y_cal

def _calibrate_one(model, X_cal, y_cal, method=CALIBRATION_METHOD):
    """Zwraca dopasowany CalibratedClassifierCV (prefit) lub sam model przy błędzie."""
    try:
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        return model  # fallback bez kalibracji

def _fit_bases_with_calibration(X_train, y_train, base_models, frac=CALIBRATION_FRAC, method=CALIBRATION_METHOD):
    """
    Fit baz (na X_fit) i kalibruj na X_cal. Zwraca:
    - dict calibrated {'rf': clf, 'xgb': clf, 'lgb': clf}
    - dict ap_on_cal  {'rf': AP,  'xgb': AP,  'lgb': AP}
    """
    X_fit, y_fit, X_cal, y_cal = _split_calibration_tail(X_train, y_train, frac=frac)

    # dopasuj bazy
    for name, mdl in base_models.items():
        mdl.fit(X_fit, y_fit)

    # kalibruj
    calibrated = {}
    ap_on_cal = {}
    for name, mdl in base_models.items():
        mdl_cal = _calibrate_one(mdl, X_cal, y_cal, method=method)

        # policz AP na kalibracji do wag głosowania
        try:
            p_cal = mdl_cal.predict_proba(X_cal)[:, 1]
            ap = average_precision_score(y_cal, p_cal) if len(y_cal) > 0 else 0.0
        except Exception:
            ap = 0.0
        calibrated[name] = mdl_cal
        ap_on_cal[name] = float(ap)

    return calibrated, ap_on_cal

def _normalize_weights_from_ap(ap_dict, fallback=VOTING_WEIGHTS):
    """Wagi ∝ AP (z kalibracji). Gdy zero/NaN, wraca fallback."""
    ws = {k: (v if np.isfinite(v) and v > 0 else 0.0) for k, v in ap_dict.items()}
    s = sum(ws.values())
    if s <= 0:
        return fallback.copy()
    return {k: ws[k] / s for k in ws.keys()}

def _soft_vote_proba(p_rf, p_xgb, p_lgb, w):
    """zwrot wagowanej średniej proba"""
    return w["rf"]*p_rf + w["xgb"]*p_xgb + w["lgb"]*p_lgb    

def walk_forward_evaluate_stacking(feat_frames, train_days, test_days, step_days):
    # lokalne importy (żeby nie modyfikować nagłówka pliku)
    from sklearn.calibration import CalibratedClassifierCV

    all_feat = pd.concat(feat_frames, axis=0, sort=False).sort_index()
    X_cols = [c for c in all_feat.columns if c not in ("y","FWD_RET","CLOSE","Ticker")]

    start_time = all_feat.index.min(); end_time = all_feat.index.max()
    t0_list = []; t0 = start_time + dt.timedelta(days=train_days)
    while True:
        train_start = t0 - dt.timedelta(days=train_days)
        train_end   = t0
        test_end    = train_end + dt.timedelta(days=test_days)
        if test_end > end_time: break
        t0_list.append(t0); t0 = t0 + dt.timedelta(days=step_days)

    window_metrics = []; overall_true = []; overall_proba = []
    last_bundle = None

    for t0 in tqdm(t0_list, desc="AI: walk-forward (stacking)", mininterval=0.2):
        train_start = t0 - dt.timedelta(days=train_days)
        train_end   = t0
        test_end    = train_end + dt.timedelta(days=test_days)

        train = all_feat.loc[(all_feat.index>train_start)&(all_feat.index<=train_end)]
        test  = all_feat.loc[(all_feat.index>train_end)&(all_feat.index<=test_end)]
        if len(train) < 200 or len(test) < 15:
            continue

        X_train, y_train = train[X_cols].copy(), train["y"].copy()
        X_test,  y_test  = test[X_cols].copy(),  test["y"].copy()
        X_train, med = sanitize_matrix(X_train, None)
        X_test, _    = sanitize_matrix(X_test, med)

        # === (1) Stacking z OOF (jak było) — daje meta-klasyfikator
        oof_meta, bundle = build_oof_and_fit_bases(X_train, y_train, n_splits=3)

        # Predykcje bazowe na teście (do meta)
        proba_rf_base  = bundle["rf"].predict_proba(X_test)[:,1]
        proba_xgb_base = bundle["xgb"].predict_proba(X_test)[:,1]
        proba_lgb_base = bundle["lgb"].predict_proba(X_test)[:,1]
        meta_X_test = np.column_stack([proba_rf_base, proba_xgb_base, proba_lgb_base])
        proba_meta = bundle["meta"].predict_proba(meta_X_test)[:,1]

        # === (2) Kalibracja baz (Platt/sigmoid, cv=3) + alternatywne blendy
        # Dopasuj świeże (niezależne) kopie bazowych modeli i skalibruj na X_train
        # (używamy tych samych ustawień co make_base_models() przez policzenie spw)
        pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
        spw = (neg / max(pos, 1))
        rf0, xgb0, lgb0 = make_base_models(spw)

        rf0.fit(X_train, y_train)
        xgb0.fit(X_train, y_train)
        lgb0.fit(X_train, y_train)

        rf_cal  = CalibratedClassifierCV(rf0,  cv=3, method="sigmoid").fit(X_train, y_train)
        xgb_cal = CalibratedClassifierCV(xgb0, cv=3, method="sigmoid").fit(X_train, y_train)
        lgb_cal = CalibratedClassifierCV(lgb0, cv=3, method="sigmoid").fit(X_train, y_train)

        p_rf  = rf_cal.predict_proba(X_test)[:,1]
        p_xgb = xgb_cal.predict_proba(X_test)[:,1]
        p_lgb = lgb_cal.predict_proba(X_test)[:,1]

        # Soft voting (średnia skalibrowanych prawdopodobieństw)
        proba_soft = (p_rf + p_xgb + p_lgb) / 3.0

        # Rank voting (średnia znormalizowanych rang)
        def _rank_avg(pvec):
            r = pd.Series(pvec).rank(method="average")  # 1..N
            return (r - 1.0) / max(len(r) - 1.0, 1.0)   # 0..1
        r_rf  = _rank_avg(p_rf)
        r_xgb = _rank_avg(p_xgb)
        r_lgb = _rank_avg(p_lgb)
        proba_rank = (r_rf + r_xgb + r_lgb) / 3.0
        proba_rank = proba_rank.values.astype(float)

        # === (3) Końcowy blend: 50% meta + 25% soft + 25% rank
        proba_ens = 0.50 * proba_meta + 0.25 * proba_soft + 0.25 * proba_rank

        # === (4) Metryki okna
        y_true    = y_test.values
        auc = roc_auc_score(y_true, proba_ens) if len(np.unique(y_true))>1 else np.nan
        ap  = average_precision_score(y_true, proba_ens) if len(y_true)>0 else np.nan
        y_pred = (proba_ens >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else np.nan
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        window_metrics.append({
            "Train_Start": train_start.date(), "Train_End": train_end.date(),
            "Test_End": test_end.date(), "N_train": len(X_train), "N_test": len(X_test),
            "AUC": auc, "AP": ap, "ACC": acc, "Precision": prec, "Recall": rec
        })

        overall_true.append(y_true)
        overall_proba.append(proba_ens)
        # Zachowujemy bundle do predykcji „predict” (meta + bazy + mediany)
        last_bundle = {"models": bundle, "X_cols": X_cols, "median_fill": med}

    if not window_metrics:
        return None, None, None, None, X_cols

    metrics_df   = pd.DataFrame(window_metrics)
    overall_true = np.concatenate(overall_true)
    overall_proba= np.concatenate(overall_proba)
    return metrics_df, overall_true, overall_proba, last_bundle, X_cols


def _compute_atr14(df_prices):
    prev = df_prices["CLOSE"].shift(1)
    tr = pd.concat([
        (df_prices["HIGH"] - df_prices["LOW"]).abs(),
        (df_prices["HIGH"] - prev).abs(),
        (df_prices["LOW"]  - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=7).mean()

def _compute_atr(df_prices, window=14):
    prev = df_prices["CLOSE"].shift(1)
    tr = pd.concat([
        (df_prices["HIGH"] - df_prices["LOW"]).abs(),
        (df_prices["HIGH"] - prev).abs(),
        (df_prices["LOW"]  - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=max(7, window//2)).mean()

def add_sltp_levels(
    df_in: pd.DataFrame,
    as_of_dt: dt.datetime,
    atr_window: int = 14,
    atr_mult_sl: float = 1.5,
    tp_mults: tuple = (1.0, 2.0),
    min_hist_days: int = 120
):
    """
    SWING: wyznacza Entry, ATR, SL (= Entry - 1.5*ATR), TP1 (= Entry + 1.0*ATR), TP2 (= Entry + 2.0*ATR)
    oraz R/R do TP1/TP2. Preferuje 'Close' z df_in jako cenę wejścia.
    """
    if df_in is None or df_in.empty or "Ticker" not in df_in.columns:
        return df_in

    end_dt = as_of_dt
    start_dt = as_of_dt - dt.timedelta(days=min_hist_days)

    out = df_in.copy()
    atr_map, entry_map = {}, {}
    sl_map, tp1_map, tp2_map = {}, {}, {}
    rr1_map, rr2_map = {}, {}

    # preferowana cena wejścia z kolumny 'Close' (jeśli jest)
    close_map = {}
    if "Close" in out.columns:
        try:
            close_map = out.set_index("Ticker")["Close"].to_dict()
        except Exception:
            close_map = {}

    for t in out["Ticker"].astype(str):
        stq = fetch_from_stooq(t)
        dfr = prepare_stooq_df(stq, start_dt, end_dt) if stq is not None else None
        if dfr is None or dfr.empty:
            atr_map[t]  = np.nan
            entry_map[t]= np.nan
            sl_map[t]   = np.nan; tp1_map[t] = np.nan; tp2_map[t] = np.nan
            rr1_map[t]  = np.nan; rr2_map[t] = np.nan
            continue

        d = dfr.copy()
        d["ATR"] = _compute_atr(d, window=atr_window)
        d = d.dropna(subset=["ATR"])
        if d.empty:
            atr_map[t]  = np.nan
            entry_map[t]= np.nan
            sl_map[t]   = np.nan; tp1_map[t] = np.nan; tp2_map[t] = np.nan
            rr1_map[t]  = np.nan; rr2_map[t] = np.nan
            continue

        atr   = float(d["ATR"].iloc[-1])
        entry = float(close_map.get(t, d["CLOSE"].iloc[-1]))

        sl   = entry - atr_mult_sl * atr
        tp1  = entry + (tp_mults[0] if len(tp_mults) >= 1 else 1.0) * atr
        tp2  = entry + (tp_mults[1] if len(tp_mults) >= 2 else tp_mults[0]) * atr
        risk = max(1e-8, entry - sl)
        rr1  = (tp1 - entry) / risk
        rr2  = (tp2 - entry) / risk

        atr_map[t]   = atr
        entry_map[t] = entry
        sl_map[t]    = sl
        tp1_map[t]   = tp1
        tp2_map[t]   = tp2
        rr1_map[t]   = rr1
        rr2_map[t]   = rr2

    out["ATR14"]  = out["Ticker"].map(atr_map).round(4)
    out["Entry"]  = out["Ticker"].map(entry_map).round(4)
    out["SL"]     = out["Ticker"].map(sl_map).round(4)
    out["TP1"]    = out["Ticker"].map(tp1_map).round(4)
    out["TP2"]    = out["Ticker"].map(tp2_map).round(4)
    out["RR_TP1"] = out["Ticker"].map(rr1_map).round(3)
    out["RR_TP2"] = out["Ticker"].map(rr2_map).round(3)
    return out


def _build_signal_message(date_str, top_buys_df, limit=10):
    lines = []
    lines.append(f"GPW — sygnały AI ({date_str})")
    if top_buys_df is None or top_buys_df.empty:
        lines.append("Brak sygnałów BUY po filtrze.")
        return "\n".join(lines)

    df = top_buys_df.head(limit).copy()
    cols = [c for c in ["Ticker","Close","AI_Proba","ATR14","SL","TP1","TP2","RR_TP1","RR_TP2"] if c in df.columns]
    lines.append(f"Top BUY ({len(df)}):")
    for _, r in df.iterrows():
        lines.append(
            f"- {r['Ticker']}: p={r.get('AI_Proba', float('nan')):.3f} | "
            f"C={r.get('Close', float('nan')):.2f} | SL={r.get('SL', float('nan')):.2f} | "
            f"TP1={r.get('TP1', float('nan')):.2f} (RR={r.get('RR_TP1', float('nan')):.2f}) | "
            f"TP2={r.get('TP2', float('nan')):.2f} (RR={r.get('RR_TP2', float('nan')):.2f})"
        )
    return "\n".join(lines)

def send_email_message(subject, body, to_addr=None, attachments=None):
    """Wysyła e-mail przez SMTP (TLS/587). attachments: lista ścieżek plików."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    to   = to_addr or os.getenv("EMAIL_TO")

    if not all([host, port, user, pwd, to]):
        print("[EMAIL] Brak konfiguracji SMTP w zmiennych środowiskowych.")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((str(Header("GPW AI Bot", "utf-8")), user))
    msg["To"] = to

    msg.attach(MIMEText(body, _charset="utf-8"))

    # Załączniki (opcjonalnie)
    attachments = attachments or []
    for path in attachments:
        try:
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            filename = os.path.basename(path)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
        except Exception as e:
            print(f"[EMAIL] Nie udało się dołączyć {path}: {e}")

    try:
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())
        print("[EMAIL] Wysłano.")
        return True
    except Exception as e:
        print("[EMAIL][ERR]", e)
        return False

def predict_today_with_ensemble(summary_df, bundle, today, idx_df, train_days=AI_TRAIN_DAYS):
    end_dt = today
    start_dt = today - dt.timedelta(days=train_days)
    rows = []

    X_cols = bundle["X_cols"]
    med    = bundle["median_fill"]
    rf     = bundle["models"]["rf"]
    xgb    = bundle["models"]["xgb"]
    lgb    = bundle["models"]["lgb"]
    meta   = bundle["models"]["meta"]

    for _, row in summary_df.iterrows():
        ticker = str(row["Ticker"]).strip()
        stq = fetch_from_stooq(ticker)
        dfr = prepare_stooq_df(stq, start_dt, end_dt) if stq is not None else None
        if dfr is None or dfr.empty: continue

        d = compute_feat_frame(dfr, idx_df=idx_df)
        if d.empty: continue

        x = d.iloc[[-1]].copy()
        x = replace_inf_with_nan(x); x = clip_extremes(x)
        x = x.fillna(med).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        missing = [c for c in X_cols if c not in x.columns]
        if missing: continue

        xb = x[X_cols]
        prf  = rf.predict_proba(xb)[:,1]
        pxgb = xgb.predict_proba(xb)[:,1]
        plgb = lgb.predict_proba(xb)[:,1]
        p    = meta.predict_proba(np.column_stack([prf, pxgb, plgb]))[:,1]

        rows.append({"Ticker": ticker, "AI_Proba": float(p[0]), "Date": d.index[-1].date()})
        time.sleep(0.005 + random.random()*0.01)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Ticker","AI_Proba","Date"])

def build_daily_signals_with_ai(summary_df, today, pred_df, ai_proba_threshold=AI_PROBA_THRESHOLD):
    end_dt=today; start_dt=today-dt.timedelta(days=LOOKBACK_DAYS); daily_rows=[]
    for _,row in summary_df.iterrows():
        ticker=str(row["Ticker"]).strip()
        mfi_lower=float(row["mfi_lower"]); mfi_upper=float(row["mfi_upper"])
        rsi_lower=float(row["rsi_lower"]); rsi_upper=float(row["rsi_upper"])
        stq=fetch_from_stooq(ticker); dfr=prepare_stooq_df(stq,start_dt,end_dt) if stq is not None else None
        if dfr is None or dfr.empty: continue
        d=dfr.copy(); d["VOL"]=d.get("VOL",0).fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d["MFI"]=MFIIndicator(high=d["HIGH"],low=d["LOW"],close=d["CLOSE"],volume=d["VOL"],window=14).money_flow_index()
            d["RSI"]=RSIIndicator(close=d["CLOSE"],window=14).rsi()
            macd=MACD(close=d["CLOSE"],window_slow=26,window_fast=12,window_sign=9)
            d["MACD"]=macd.macd(); d["MACD_signal"]=macd.macd_signal()
            d["SMA10"]=d["CLOSE"].rolling(10,min_periods=5).mean()
            d["SMA30"]=d["CLOSE"].rolling(30,min_periods=10).mean()
        d=d.dropna(subset=["MFI","RSI","MACD","MACD_signal","SMA10","SMA30"])
        if d.empty: continue
        latest=d.iloc[-1]
        macd_trend="wzrostowy" if latest["MACD"]>=latest["MACD_signal"] else "spadkowy"
        sma_trend ="wzrostowy" if latest["SMA10"]>=latest["SMA30"] else "spadkowy"
        if (latest["MFI"]<mfi_lower) and (latest["RSI"]<rsi_lower):
            signal="buy"
        elif (latest["MFI"]>mfi_upper) or (latest["RSI"]>rsi_upper):
            signal="sell"
        else:
            signal="hold"
        daily_rows.append({
            "Ticker":ticker,"Date":d.index[-1].date(),"Close":float(latest["CLOSE"]),
            "MFI":float(latest["MFI"]),"RSI":float(latest["RSI"]),
            "MACD_Trend":macd_trend,"SMA_Trend":sma_trend,"Signal":signal
        })
    daily_df=pd.DataFrame(daily_rows)
    merged=daily_df.merge(pred_df[["Ticker","AI_Proba"]],on="Ticker",how="left")

    def ai_filter(r):
        if r["Signal"] == "buy" and r.get("AI_Proba", np.nan) >= ai_proba_threshold: return "buy"
        if r["Signal"] == "sell": return "sell"
        return "hold"

    merged["AI_Filtered_Signal"]=merged.apply(ai_filter,axis=1)
    return merged

def save_text_summary_pdf(metrics_df, topN_df, topN_filtered_df, today, out_pdf_path, thr=None):
    lines=[]
    # Nagłówek
    lines.append("Raport dzienny AI (skrót)")
    lines.append(f"Data: {today.strftime('%Y-%m-%d')}")
    if thr is not None:
        lines.append(f"Próg AI_Proba (thr): {thr:.3f}")
    lines.append("")

    # Metryki WF (jeśli są)
    if metrics_df is not None and not metrics_df.empty:
        med_auc = metrics_df["AUC"].median(skipna=True)
        med_ap  = metrics_df["AP"].median(skipna=True)
        mean_acc= metrics_df["ACC"].mean(skipna=True)
        wins = len(metrics_df)
        lines += [
            f"Okna walk-forward: {wins}",
            f"Mediana AUC: {med_auc:.3f}" if pd.notnull(med_auc) else "Mediana AUC: n/a",
            f"Mediana AP:  {med_ap:.3f}" if pd.notnull(med_ap) else "Mediana AP: n/a",
            f"Średnie ACC: {mean_acc:.3f}" if pd.notnull(mean_acc) else "Średnie ACC: n/a",
            ""
        ]
    else:
        lines += ["Brak metryk walk-forward.", ""]

    # Top wg AI_Proba (lista)
    lines.append("10 najmocniejszych sygnałów:")
    if topN_df is not None and not topN_df.empty:
        for i, r in topN_df.iterrows():
            lines.append(f"{i+1}. {r['Ticker']}: AI_Proba={r.get('AI_Proba', float('nan')):.3f}, "
                         f"Signal={r.get('Signal', 'n/a')}")
    else:
        lines.append("(brak)")
    lines.append("")

    # Sekcja 1 (jak dotąd): szczegóły BUY + poziomy
    lines.append("Najmocniejsze sygnały zakupu >= progu AI")
    if topN_filtered_df is not None and not topN_filtered_df.empty:
        header = f"{'#':>2} {'Ticker':<8} {'Close':>9} {'ATR14':>8} {'SL':>9} {'TP1':>9} {'TP2':>9} {'RR1':>6} {'RR2':>6}"
        lines.append(header)
        lines.append("-"*len(header))
        for i, r in topN_filtered_df.reset_index(drop=True).iterrows():
            lines.append(
                f"{i+1:>2} {str(r['Ticker']):<8} "
                f"{r.get('Close', float('nan')):>9.4f} {r.get('ATR14', float('nan')):>8.4f} "
                f"{r.get('SL', float('nan')):>9.4f} {r.get('TP1', float('nan')):>9.4f} {r.get('TP2', float('nan')):>9.4f} "
                f"{r.get('RR_TP1', float('nan')):>6.2f} {r.get('RR_TP2', float('nan')):>6.2f}"
            )
    else:
        lines.append("(brak)")
    lines.append("")

    # Render do PDF
    with PdfPages(out_pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        plt.axis("off")
        # Dziel długi tekst na strony jeśli kiedyś bardzo urośnie
        text = "\n".join(lines)
        plt.text(0.05, 0.95, text, fontsize=10, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

def save_bundle(bundle, threshold, outpath: Path):
    """Zapisz modele + median_fill (joblib) oraz metadane (JSON)."""
    meta = {
        "X_cols": bundle["X_cols"],
        "threshold": float(threshold)
    }
    dump(
        {"models": bundle["models"], "X_cols": bundle["X_cols"], "median_fill": bundle["median_fill"]},
        outpath.with_suffix(".joblib")
    )
    (outpath.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))

def load_bundle(inpath: Path):
    """Wczytaj modele i próg z plików out/models/<tag>.joblib + .json."""
    obj = load(inpath.with_suffix(".joblib"))
    meta = json.loads((inpath.with_suffix(".json")).read_text())
    bundle = {"models": obj["models"], "X_cols": obj["X_cols"], "median_fill": obj["median_fill"]}
    return bundle, float(meta.get("threshold", 0.5))

def save_trained_bundle(bundle: dict, X_cols, median_fill, today):
    """Zapisuje modele i metadane do folderu OUT_DIR."""
    ts = today.strftime("%Y%m%d_%H%M%S")

    # Metadane (kolumny, mediany, wersje)
    meta = {
        "timestamp": ts,
        "X_cols": list(X_cols),
        "median_fill": (median_fill.to_dict() if hasattr(median_fill, "to_dict") else None),
        "notes": "GPW walk-forward stacking bundle"
    }

    # Zapis modeli (joblib)
    paths = {}
    for name in ["rf", "xgb", "lgb", "meta"]:
        model = bundle.get(name)
        if model is None:
            continue
        path = os.path.join(OUT_DIR, f"model_{name}_{ts}.joblib")
        dump(model, path)
        paths[name] = path

    # Manifest z metadanymi i ścieżkami
    manifest = {
        "models": paths,
        "meta": meta,
        "type": "stacking_bundle"
    }
    manifest_path = os.path.join(OUT_DIR, f"bundle_manifest_{ts}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Zapisano modele: {paths}")
    print(f"[SAVE] Manifest: {manifest_path}")


# ===================== MAIN =====================
def main():
    # === NOWE: CLI mode ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["train", "predict"], default="train",
        help="train = pełny WF + zapis modelu; predict = tylko dzienne predykcje z ostatniego modelu"
    )
    args = parser.parse_args()

    today = dt.datetime.now()
    end_dt = today

        # === NOWE: tryb PREDICT (bez trenowania) ===
        # === NOWE: tryb PREDICT (bez trenowania) ===
    if args.mode == "predict":
        model_tag = "latest"
        model_path = MODEL_DIR / model_tag
        if not (model_path.with_suffix(".joblib")).exists():
            print("Brak modelu w out/models/latest.joblib — uruchom najpierw --mode train.")
            return
    
        # Wczytaj paczkę modeli + próg
        bundle, thr_star = load_bundle(model_path)
    
        # Potrzebny ostatni 'wyniki_*.xlsx' aby mieć listę tickerów i progi MFI/RSI
        import glob as _glob
        cand = sorted(_glob.glob("wyniki_*.xlsx")) + sorted([str(p) for p in OUT_DIR.glob("wyniki_*.xlsx")])
        if not cand:
            print("Nie znaleziono pliku wyniki_*.xlsx. Uruchom najpierw --mode train.")
            return
        latest_xlsx = cand[-1]
        summary_df = pd.read_excel(latest_xlsx, sheet_name="Podsumowanie")
    
        # Predykcje na dziś
        idx_today = fetch_wig20_frame(today - dt.timedelta(days=AI_TRAIN_DAYS), today)
        pred_df = predict_today_with_ensemble(summary_df, bundle, today, idx_today, train_days=AI_TRAIN_DAYS)
        merged = build_daily_signals_with_ai(summary_df, today, pred_df, ai_proba_threshold=thr_star)
    
        # Zapis xlsx z dziennymi sygnałami i TOP-N
        out_ai = RUN_DIR / f"sygnaly_{today.strftime('%Y%m%d')}_ai.xlsx"
    
        # 1) Przygotuj TOP-y
        topN_df = merged.dropna(subset=["AI_Proba"])\
                        .sort_values("AI_Proba", ascending=False)\
                        .head(TOP_N).reset_index(drop=True)
    
        topN_filtered_df = merged.query("AI_Filtered_Signal == 'buy'")\
                                 .dropna(subset=["AI_Proba"])\
                                 .sort_values("AI_Proba", ascending=False)\
                                 .head(TOP_N).reset_index(drop=True)
    
        # 2) Dodaj poziomy SL/TP PRZED zapisem do Excela
        topN_df = add_sltp_levels(topN_df, as_of_dt=today, atr_mult_sl=1.5, tp_mults=(1.0, 2.0))
        topN_filtered_df = add_sltp_levels(topN_filtered_df, as_of_dt=today, atr_mult_sl=1.5, tp_mults=(1.0, 2.0))
    
        # 3) Zapisz wszystkie arkusze jednym writerem i zamknij plik
        with pd.ExcelWriter(out_ai, engine="openpyxl") as w:
            merged.to_excel(w, sheet_name="Daily_Signals_AI", index=False)
            topN_df.to_excel(w, sheet_name=f"Top{TOP_N}_By_AI_Proba", index=False)
            topN_filtered_df.to_excel(w, sheet_name=f"Top{TOP_N}_Filtered_Buys", index=False)
        print("OK. (predict) Zapisano sygnały do", out_ai)
    
        # 4) (Opcjonalnie) wczytaj metryki WF do PDF
        metrics_df_predict = None
        try:
            stats_cand = sorted(_glob.glob("model_stats_*.xlsx")) \
                       + sorted([str(p) for p in OUT_DIR.glob("model_stats_*.xlsx")])
            if stats_cand:
                latest_stats = stats_cand[-1]
                metrics_df_predict = pd.read_excel(latest_stats, sheet_name="Model_Stats")
                print("Załadowano metryki z:", latest_stats)
            else:
                print("Nie znaleziono model_stats_*.xlsx – raport pokaże 'Brak metryk walk-forward.'")
        except Exception as e:
            print("Błąd wczytywania metryk:", e)
    
        # 5) Wygeneruj PDF podsumowania (już po zapisie Excela) -> uwzględnia SL/TP
        summary_pdf = RUN_DIR / f"model_summary_{today.strftime('%Y%m%d')}.pdf"
        try:
            save_text_summary_pdf(
                metrics_df=metrics_df_predict,
                topN_df=topN_df,
                topN_filtered_df=topN_filtered_df,
                today=today,
                out_pdf_path=summary_pdf,
                thr=thr_star
            )
            print("OK. (predict) Zapisano raport PDF:", summary_pdf)
        except Exception as e:
            print("[PREDICT][ERROR] Nie udało się wygenerować model_summary_*.pdf:", e)
    
        # 6) Wyślij maila (po zamknięciu wszystkich plików)
        date_str = today.strftime("%Y-%m-%d")
        email_subject = f"Sygnały GPW AI — {date_str}"
        email_body = _build_signal_message(date_str, topN_filtered_df, limit=TOP_N)
        send_email_message(
            email_subject,
            email_body,
            attachments=[str(out_ai), str(summary_pdf)]
        )

    sys.exit(0)   # kończymy po predykcji

    today = dt.datetime.now()
    end_dt = today

    # --- Prefilter: ostatnie 180 SESJI (+5 roboczych buforu na święta/urwane sesje) ---
    start_pref = (pd.Timestamp(today) - BDay(LIQ_LOOKBACK_DAYS + 5)).to_pydatetime()

    # --- Backtest MFI/RSI jak dotąd: 90 dni kalendarzowych ---
    start_bt = today - dt.timedelta(days=LOOKBACK_DAYS)

    # tickery
    tickers=fetch_bankier_tickers()
    print("Pobrane tickery (%d):"%len(tickers), tickers)
    if not tickers:
        print("Brak tickerów – koniec."); return

    # prefilter (płynność/ruch)
    kept, liq_df, th = prefilter_tickers_auto(
        tickers, start_pref, end_dt,
        fetch_fn=fetch_from_stooq,
        prep_fn=prepare_stooq_df,
        mode=PREFILTER_MODE
    )
    tickers = kept

    # backtest prosty MFI/RSI + wybór najlepszych progów
    results=[]; signals=[]
    for t in tqdm(tickers, desc="Backtest MFI/RSI (Stooq)"):
        stq=fetch_from_stooq(t)
        df_raw = prepare_stooq_df(stq, start_bt, end_dt) if stq is not None else None
        out=process_ticker(t, df_raw)
        if out is None: continue
        results.append({k:v for k,v in out.items() if k!="signals_df"})
        signals.append(out["signals_df"])
        time.sleep(0.04+random.random()*0.06)
    if not results:
        print("Brak wyników – za mało danych.")
        return
    summary_df=pd.DataFrame(results).sort_values("Signal_Score(%)",ascending=False).reset_index(drop=True)
    signals_df=pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=["Date","Signal","Ticker"])

    out_name=f"wyniki_{today.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(out_name, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Podsumowanie", index=False)
        signals_df.to_excel(w, sheet_name="Sygnały", index=False)
    print("OK. Zapisano:", out_name)

    # AI: zbuduj cechy (bez fundamentów), etykiety, train walk-forward
    ai_end=today; ai_start=today-dt.timedelta(days=AI_TRAIN_DAYS+AI_TEST_DAYS+200)
    idx_df=fetch_wig20_frame(ai_start, ai_end)

    feat_frames=[]
    for t in tqdm(summary_df["Ticker"].tolist(), desc="AI: cechy + etykiety"):
        stq=fetch_from_stooq(str(t))
        dfr=prepare_stooq_df(stq, ai_start, ai_end) if stq is not None else None
        if dfr is None or dfr.empty: continue
        d=compute_feat_frame(dfr, idx_df=idx_df)
        d=make_labels(d)
        if d.empty: continue
        d["Ticker"]=str(t)
        feat_frames.append(d)
        time.sleep(0.02+random.random()*0.03)

    if not feat_frames:
        print("AI: brak danych do walk-forward.")
        return out_name, summary_df, signals_df

    metrics_df, overall_true, overall_proba, last_bundle, X_cols = walk_forward_evaluate_stacking(
        feat_frames, AI_TRAIN_DAYS, AI_TEST_DAYS, AI_STEP_DAYS
    )

    if metrics_df is None:
        print("AI: WF nie wyszedł — fallback 80/20.")
        all_feat=pd.concat(feat_frames,axis=0,sort=False).sort_index()
        X_cols=[c for c in all_feat.columns if c not in ("y","FWD_RET","CLOSE","Ticker")]
        cutoff=int(len(all_feat)*0.8)
        train=all_feat.iloc[:cutoff].copy()
        valid=all_feat.iloc[cutoff:].copy()
        X_train,med=sanitize_matrix(train[X_cols],None)
        valid_X,_=sanitize_matrix(valid[X_cols],med)
        if IMBLEARN_OK:
            try:
                X_train,y_train=SMOTEENN(random_state=RANDOM_STATE).fit_resample(X_train,train["y"])
            except Exception:
                y_train=train["y"]
        else:
            y_train=train["y"]
        clf=RandomForestClassifier(
            n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,random_state=RANDOM_STATE,
            class_weight=None if IMBLEARN_OK else CLASS_WEIGHT,n_jobs=-1
        )
        clf.fit(X_train,y_train); last_model=clf
        try:
            proba_valid=clf.predict_proba(valid_X)[:,1]
            overall_true=valid["y"].values; overall_proba=proba_valid
            metrics_df=pd.DataFrame([{
                "Train_Start":train.index.min().date(),"Train_End":train.index.max().date(),
                "Test_End":valid.index.max().date(),"N_train":len(X_train),"N_test":len(valid),
                "AUC":roc_auc_score(valid["y"],proba_valid) if len(valid)>0 else np.nan,
                "AP":average_precision_score(valid["y"],proba_valid) if len(valid)>0 else np.nan,
                "ACC":accuracy_score(valid["y"],(proba_valid>=0.5).astype(int)) if len(valid)>0 else np.nan,
                "Precision":precision_score(valid["y"],(proba_valid>=0.5).astype(int),zero_division=0) if len(valid)>0 else 0,
                "Recall":recall_score(valid["y"],(proba_valid>=0.5).astype(int),zero_division=0) if len(valid)>0 else 0
            }])
        except Exception:
            overall_true,overall_proba=np.array([]),np.array([])
    # === Rekomendacja progu na podstawie krzywej Precision-Recall ===
    thr_star, p_at, r_at = 0.50, np.nan, np.nan
    if overall_true is not None and overall_proba is not None \
       and len(overall_true) > 0 and len(overall_proba) > 0:
        try:
            thr_star, p_at, r_at = pick_threshold_for_precision(overall_proba, overall_true, min_precision=0.70)
            print(f"Recommended AI_PROBA_THRESHOLD = {thr_star:.3f} (Precision={p_at:.3f}, Recall={r_at:.3f})")
        except Exception as e:
            print("Threshold pick failed:", e)
    else:
        print("Brak surowych predykcji do wyznaczenia progu – używam fallback 0.50.")

    # zapisz metryki i FI
    stats_name = f"model_stats_{today.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(stats_name, engine="openpyxl") as w:
        metrics_df.to_excel(w, sheet_name="Model_Stats", index=False)
        try:
            fi = pd.DataFrame({"Feature": X_cols, "Importance": last_model.feature_importances_}) \
                .sort_values("Importance", ascending=False).reset_index(drop=True)
            fi.to_excel(w, sheet_name="Feature_Importance", index=False)
        except Exception:
            pass
        # --- NOWY ARKUSZ: wagi meta-modelu (stacking) ---
        try:
            if last_bundle is not None and "models" in last_bundle and "meta" in last_bundle["models"]:
                meta = last_bundle["models"]["meta"]
                coef = getattr(meta, "coef_", None)
                if coef is not None:
                    meta_df = pd.DataFrame(
                        {"Base_Model": ["rf", "xgb", "lgb"], "Meta_Weight": coef.ravel().tolist()}
                    )
                    meta_df.to_excel(w, sheet_name="Meta_Coefficients", index=False)
        except Exception:
            pass

        # --- NOWY ARKUSZ: rekomendacja progu ---
        pd.DataFrame([{
            "objective": "Max Recall s.t. Precision >= 0.70",
            "recommended_threshold": round(thr_star, 3),
            "Precision_at_thr": (round(p_at, 3) if pd.notnull(p_at) else "n/a"),
            "Recall_at_thr": (round(r_at, 3) if pd.notnull(r_at) else "n/a")
        }]).to_excel(w, sheet_name="Threshold_Recommendation", index=False)

    print("OK. Zapisano metryki:", stats_name)

	    # --- ZAPIS MODELU DO out/models/latest.joblib + latest.json ---
    try:
        if last_bundle is not None and "models" in last_bundle:
            # zapis paczki do przewidywań (to na tym polega --mode predict)
            save_bundle(last_bundle, thr_star, MODEL_DIR / "latest")
            print("[SAVE] Zapisano paczkę do out/models/latest.joblib (+ .json)")

            # dodatkowo: archiwalne kopie do OUT_DIR/model_*.joblib (opcjonalnie)
            save_trained_bundle(
                bundle=last_bundle["models"],
                X_cols=X_cols,
                median_fill=last_bundle.get("median_fill", None),
                today=today
            )
        else:
            print("[SAVE] Brak last_bundle/models – nic do zapisania.")
    except Exception as e:
        print("[SAVE][ERROR] Nie udało się zapisać paczki modeli:", e)

    # wykresy ROC/PR
    plots_name=f"model_plots_{today.strftime('%Y%m%d')}.pdf"
    if overall_true is not None and len(overall_true)>0:
        save_plots_pdf(overall_true, overall_proba, None, None, out_pdf_path=plots_name)
    else:
        with PdfPages(plots_name) as pdf:
            fig=plt.figure(); plt.axis("off"); plt.text(0.1,0.9,"Brak danych do ROC/PR",fontsize=12,va="top"); pdf.savefig(fig); plt.close(fig)
    print("OK. Zapisano wykresy:", plots_name)

    # predykcje na dziś + dzienne sygnały
    idx_today=fetch_wig20_frame(today-dt.timedelta(days=AI_TRAIN_DAYS), today)
    pred_df = predict_today_with_ensemble(summary_df, last_bundle, today, idx_today, train_days=AI_TRAIN_DAYS)
    merged = build_daily_signals_with_ai(summary_df, today, pred_df, ai_proba_threshold=thr_star)
    topN_df=merged.dropna(subset=["AI_Proba"]).sort_values("AI_Proba",ascending=False).head(TOP_N).reset_index(drop=True)
    topN_filtered_df=merged.query("AI_Filtered_Signal == 'buy'").dropna(subset=["AI_Proba"]).sort_values("AI_Proba",ascending=False).head(TOP_N).reset_index(drop=True)

    out_ai=f"sygnaly_{today.strftime('%Y%m%d')}_ai.xlsx"
    with pd.ExcelWriter(out_ai, engine="openpyxl") as w:
        merged.to_excel(w, sheet_name="Daily_Signals_AI", index=False)
        topN_df.to_excel(w, sheet_name=f"Top{TOP_N}_By_AI_Proba", index=False)
        topN_filtered_df.to_excel(w, sheet_name=f"Top{TOP_N}_Filtered_Buys", index=False)
    print("OK. Zapisano dzienne sygnały AI + TOP-N:", out_ai)

#    summary_pdf=f"model_summary_{today.strftime('%Y%m%d')}.pdf"
#    save_text_summary_pdf(metrics_df, topN_df, topN_filtered_df, today, summary_pdf)
#    print("OK. Zapisano raport PDF:", summary_pdf)

    return out_name, summary_df, signals_df, stats_name, plots_name, out_ai

if __name__ == "__main__":
    main()
