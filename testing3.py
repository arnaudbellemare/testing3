import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, t
from scipy.optimize import minimize
from numba import njit
from numpy.typing import NDArray
from typing import Optional
import streamlit as st

###############################################################################
# SESSION STATE & SIDEBAR INPUTS
###############################################################################
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"

st.title("CNO Dashboard")
st.write(f"Welcome, {st.session_state['username']}!")

# Sidebar inputs
lookback_options = {
    "1 Day": 1440,
    "3 Days": 4320,
    "1 Week": 10080,
    "2 Weeks": 20160,
    "1 Month": 43200
}
global_lookback_label = st.sidebar.selectbox(
    "Select Global Lookback Period",
    list(lookback_options.keys()),
    key="global_lookback_label"
)
global_lookback_minutes = lookback_options[global_lookback_label]
timeframe = st.sidebar.selectbox(
    "Select Timeframe", ["1m", "5m", "15m", "1h"],
    key="timeframe_widget"
)
# Change option from ADC to ACD to represent Autoregressive Conditional Duration
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Hawkes BVC", "ACD", "ACI", "BSACD1"],
    key="analysis_type"
)

###############################################################################
# 1) FETCH DATA FUNCTION
###############################################################################
def fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_ohlcv = []
    since = cutoff_ts
    max_limit = 1440
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        if last_timestamp <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_timestamp + 1
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

###############################################################################
# 2) HELPER FUNCTIONS (EMA and accumulated candle index)
###############################################################################
@njit(cache=True)
def ema(arr_in: NDArray, window: int, alpha: Optional[float] = 0) -> NDArray:
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    n = arr_in.size
    ewma = np.empty(n, dtype=np.float64)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = (arr_in[i] * alpha) + (ewma[i-1] * (1 - alpha))
    return ewma

@njit(cache=True)
def accumulated_candle_index(klines: NDArray, lookback: int = 20) -> NDArray:
    n = len(klines)
    aci = np.zeros(n, dtype=np.float64)
    if n < 2:
        return aci
    for i in range(1, n):
        open_price = klines[i, 1]
        high_price = klines[i, 2]
        low_price = klines[i, 3]
        close_price = klines[i, 4]
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        if close_price > open_price:
            candle_score = body_size + 0.5 * upper_shadow - 0.3 * lower_shadow
        else:
            candle_score = -body_size - 0.5 * lower_shadow + 0.3 * upper_shadow
        if i >= lookback:
            recent_highs = klines[i-lookback:i, 2]
            recent_lows = klines[i-lookback:i, 3]
            recent_range = np.mean(recent_highs - recent_lows)
            if recent_range > 0:
                candle_score = candle_score / recent_range
        if i > 1:
            aci[i] = 0.8 * aci[i-1] + candle_score
        else:
            aci[i] = candle_score
    return aci

###############################################################################
# 3) BS DISTRIBUTION FUNCTIONS (for BSACD1)
###############################################################################
def bs_pdf(x, kappa, sigma):
    if x <= 0:
        return 0
    term = (np.sqrt(x/sigma) - np.sqrt(sigma/x))**2
    return 1/(2 * kappa * x * np.sqrt(2*np.pi)) * (np.sqrt(x/sigma)+np.sqrt(sigma/x)) * np.exp(-term/(2*kappa**2))

def bs_logpdf(x, kappa, sigma):
    if x <= 0:
        return -np.inf
    term = (np.sqrt(x/sigma) - np.sqrt(sigma/x))**2
    return -np.log(2*kappa*x*np.sqrt(2*np.pi)) + np.log(np.sqrt(x/sigma)+np.sqrt(sigma/x)) - term/(2*kappa**2)

###############################################################################
# 4) BSACD1 MODEL (Mean-based) FUNCTIONS
###############################################################################
def bsacd1_negloglik(params, X):
    beta0, alpha, beta, tau = params
    n = len(X)
    mu = np.empty(n)
    mu[0] = np.median(X)
    neglog = 0.0
    for i in range(1, n):
        mu[i] = np.exp(beta0 + alpha * np.log(mu[i-1]) + beta * (X[i-1]/mu[i-1]))
        ratio = X[i] / mu[i]
        ll = bs_logpdf(ratio, tau, 1) - np.log(mu[i])
        neglog -= ll
    return neglog

def compute_fitted_mu(params, X):
    beta0, alpha, beta, tau = params
    n = len(X)
    mu = np.empty(n)
    mu[0] = np.median(X)
    for i in range(1, n):
        mu[i] = np.exp(beta0 + alpha * np.log(mu[i-1]) + beta * (X[i-1]/mu[i-1]))
    return mu

###############################################################################
# 5) LINEAR ACD INDICATOR (ACD(1,1) Model)
###############################################################################
class LinearACDIndicator:
    def __init__(self, omega=1.0, alpha=0.5, beta=0.3):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy().sort_values("stamp")
        # Compute durations (in seconds) between successive timestamps
        df["duration"] = df["stamp"].diff().dt.total_seconds()
        durations = df["duration"].dropna().values  # length = N - 1
        if len(durations) == 0:
            return pd.DataFrame({"stamp": [], "acd": []})
        psi = np.empty(len(durations))
        psi[0] = np.median(durations)
        for i in range(1, len(durations)):
            psi[i] = self.omega + self.alpha * durations[i-1] + self.beta * psi[i-1]
        # Scale the indicator if needed
        if np.max(np.abs(psi)) != 0:
            psi = psi / np.max(np.abs(psi)) * scale
        indicator_df = pd.DataFrame({
            "stamp": df["stamp"].iloc[1:].reset_index(drop=True),
            "acd": psi
        })
        return indicator_df

###############################################################################
# 6) INDICATOR CLASSES (HawkesBVC, ACDBVC, ACIBVC)
###############################################################################
class HawkesBVC:
    def __init__(self, window=20, kappa=0.1, dof=0.25):
        self.window = window
        self.kappa = kappa
        self.dof = dof

    def _label(self, r, sigma):
        if sigma > 0.0:
            return 2 * t.cdf(r/sigma, df=self.dof) - 1.0
        else:
            return 0.0

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy().sort_values("stamp")
        prices = df["close"]
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)
        volume = df["volume"]
        sigma = r.rolling(self.window).std().fillna(0.0)
        alpha_exp = np.exp(-self.kappa)
        labels = np.array([self._label(r.iloc[i], sigma.iloc[i]) for i in range(len(r))])
        bvc = np.zeros(len(volume), dtype=float)
        current_bvc = 0.0
        for i in range(len(volume)):
            current_bvc = current_bvc * alpha_exp + volume.values[i] * labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        return pd.DataFrame({"stamp": df["stamp"], "bvc": bvc})

class ACDBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64) // 10**9
        df["duration"] = df["time_s"].diff().shift(-1)
        df = df.dropna(subset=["duration"])
        df = df[df["duration"] > 0]
        if len(df) < 10:
            return pd.DataFrame({"stamp": [], "bvc": []})
        mean_dur = df["duration"].mean()
        std_dur = df["duration"].std() or 1e-10
        df["std_resid"] = (df["duration"] - mean_dur) / std_dur
        df["price_change"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["label"] = -df["std_resid"] * df["price_change"]
        df["weighted_volume"] = df["volume"] * df["label"]
        alpha_exp = np.exp(-self.kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df["weighted_volume"].values:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        df["bvc"] = bvc
        return df[["stamp", "bvc"]].copy()

class ACIBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def estimate_intensity(self, times, beta):
        intensities = [0.0]
        for i in range(1, len(times)):
            delta_t = times[i] - times[i-1]
            intensities.append(intensities[-1] * np.exp(-beta * delta_t) + 1)
        return np.array(intensities)

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64) // 10**9
        times = df["time_s"].values
        intensities = self.estimate_intensity(times, self.kappa)
        df = df.iloc[:len(intensities)]
        df["intensity"] = intensities
        df["price_change"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["label"] = df["intensity"] * df["price_change"]
        df["weighted_volume"] = df["volume"] * df["label"]
        alpha_exp = np.exp(-self.kappa)
        bvc_list = []
        current_bvc = 0.0
        for wv in df["weighted_volume"].values:
            current_bvc = current_bvc * alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc = np.array(bvc_list)
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        df["bvc"] = bvc
        return df[["stamp", "bvc"]].copy()

###############################################################################
# 7) TUNING FUNCTIONS (using correlation with log returns)
###############################################################################
def tune_kappa_hawkes(df_prices, kappa_grid=None, scale=1e4):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa = None
    best_score = -999.0
    for k in kappa_grid:
        model = HawkesBVC(window=20, kappa=k)
        bvc_metrics = model.eval(df_prices, scale=scale)
        merged = df_prices.merge(bvc_metrics, on="stamp", how="inner")
        corr_val = merged[["log_return", "bvc"]].corr().iloc[0, 1]
        if corr_val > best_score:
            best_score = corr_val
            best_kappa = k
    return best_kappa, best_score

def tune_kappa_acd(df_prices, kappa_grid=None, scale=1e4):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa = None
    best_score = -999.0
    for k in kappa_grid:
        # Use our Linear ACD Indicator
        indicator_df = LinearACDIndicator().eval(df_prices, scale=scale)
        merged = df_prices.merge(indicator_df, left_on="stamp", right_on="stamp", how="inner")
        corr_val = merged[["log_return", "acd"]].corr().iloc[0, 1]
        if corr_val > best_score:
            best_score = corr_val
            best_kappa = k
    return best_kappa, best_score

def tune_kappa_aci(df_prices, kappa_grid=None, scale=1e5):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa = None
    best_score = -999.0
    for k in kappa_grid:
        model = ACIBVC(kappa=k)
        bvc_metrics = model.eval(df_prices, scale=scale)
        if len(bvc_metrics) == 0:
            continue
        merged = df_prices.merge(bvc_metrics, on="stamp", how="inner")
        corr_val = merged[["log_return", "bvc"]].corr().iloc[0, 1]
        if corr_val > best_score:
            best_score = corr_val
            best_kappa = k
    return best_kappa, best_score

###############################################################################
# 8) LINEAR ACD INDICATOR (ACD(1,1) Model)
###############################################################################
class LinearACDIndicator:
    def __init__(self, omega=1.0, alpha=0.5, beta=0.3):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def eval(self, df: pd.DataFrame, scale=1e4):
        df = df.copy().sort_values("stamp")
        # Compute durations (in seconds) between successive timestamps
        df["duration"] = df["stamp"].diff().dt.total_seconds()
        durations = df["duration"].dropna().values  # length = N - 1
        if len(durations) == 0:
            return pd.DataFrame({"stamp": [], "acd": []})
        psi = np.empty(len(durations))
        psi[0] = np.median(durations)
        for i in range(1, len(durations)):
            psi[i] = self.omega + self.alpha * durations[i-1] + self.beta * psi[i-1]
        if np.max(np.abs(psi)) != 0:
            psi = psi / np.max(np.abs(psi)) * scale
        indicator_df = pd.DataFrame({
            "stamp": df["stamp"].iloc[1:].reset_index(drop=True),
            "acd": psi
        })
        return indicator_df

###############################################################################
# 9) BSACD1 MODEL (Mean-based) FUNCTIONS
###############################################################################
def bsacd1_negloglik(params, X):
    beta0, alpha, beta, tau = params
    n = len(X)
    mu = np.empty(n)
    mu[0] = np.median(X)
    neglog = 0.0
    for i in range(1, n):
        mu[i] = np.exp(beta0 + alpha * np.log(mu[i-1]) + beta * (X[i-1]/mu[i-1]))
        ratio = X[i] / mu[i]
        ll = bs_logpdf(ratio, tau, 1) - np.log(mu[i])
        neglog -= ll
    return neglog

def compute_fitted_mu(params, X):
    beta0, alpha, beta, tau = params
    n = len(X)
    mu = np.empty(n)
    mu[0] = np.median(X)
    for i in range(1, n):
        mu[i] = np.exp(beta0 + alpha * np.log(mu[i-1]) + beta * (X[i-1]/mu[i-1]))
    return mu

###############################################################################
# 10) MAIN SCRIPT (STREAMLIT APP)
###############################################################################
st.header("Price & Indicator Analysis")

# Fetch data (last 12 hours)
df = fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=720)
df = df.sort_values("stamp").reset_index(drop=True)
st.write("Data range:", df["stamp"].min(), "to", df["stamp"].max())
st.write("Number of rows:", len(df))

# Compute additional fields for plotting
df["ScaledPrice"] = np.log(df["close"] / df["close"].iloc[0]) * 1e4
df["ScaledPrice_EMA"] = ema(df["ScaledPrice"].values, window=10)
df["cum_vol"] = df["volume"].cumsum()
df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
df["vwap"] = df["cum_pv"] / df["cum_vol"]
if df["vwap"].iloc[0] == 0 or not np.isfinite(df["vwap"].iloc[0]):
    df["vwap_transformed"] = df["ScaledPrice"]
else:
    df["vwap_transformed"] = np.log(df["vwap"] / df["vwap"].iloc[0]) * 1e4

# Tune parameters for each indicator
best_kappa_hawkes, best_score_hawkes = tune_kappa_hawkes(df, kappa_grid=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
st.write("Best Hawkes kappa:", best_kappa_hawkes, "with correlation:", best_score_hawkes)

best_kappa_acd, best_score_acd = tune_kappa_acd(df, kappa_grid=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])
st.write("Best ACD (duration indicator) kappa:", best_kappa_acd, "with correlation:", best_score_acd)

best_kappa_aci, best_score_aci = tune_kappa_aci(df, kappa_grid=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])
st.write("Best ACI kappa:", best_kappa_aci, "with correlation:", best_score_aci)

# ---------------------------------------------------------------------------
# Indicator Evaluation: Select based on analysis_type
# ---------------------------------------------------------------------------
if analysis_type == "Hawkes BVC":
    st.write("### Hawkes BVC Analysis")
    indicator_title = "BVC"
    indicator_df = HawkesBVC(window=20, kappa=best_kappa_hawkes).eval(df, scale=1e4)
elif analysis_type == "ACD":
    st.write("### ACD (Autoregressive Conditional Duration) Analysis")
    indicator_title = "ACD"
    # Use the Linear ACD model (ACD(1,1))
    omega = st.slider("omega", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    alpha_par = st.slider("alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    beta_par = st.slider("beta", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    indicator_df = LinearACDIndicator(omega=omega, alpha=alpha_par, beta=beta_par).eval(df, scale=1e4)
elif analysis_type == "ACI":
    st.write("### Accumulated Candle Index (ACI) Analysis")
    indicator_title = "ACI"
    df_temp = df.copy().sort_values("stamp")
    klines = df_temp[["timestamp", "open", "high", "low", "close", "volume"]].values
    aci_vals = accumulated_candle_index(klines, lookback=20)
    if np.max(np.abs(aci_vals)) != 0:
        aci_vals = aci_vals / np.max(np.abs(aci_vals)) * 1e4
    indicator_df = pd.DataFrame({"stamp": df_temp["stamp"], "bvc": aci_vals})
elif analysis_type == "BSACD1":
    st.write("### BSACD1 Model Estimation on Durations")
    df_reset = df.sort_values("stamp").reset_index(drop=True)
    df_reset["duration"] = df_reset["stamp"].diff().dt.total_seconds()
    durations = df_reset["duration"].dropna().values  # length = N-1
    init_params = [0.0, 0.5, 0.0, 1.0]
    res = minimize(bsacd1_negloglik, init_params, args=(durations,), method="L-BFGS-B")
    st.write("Estimated BSACD1 parameters:", res.x)
    fitted_mu = compute_fitted_mu(res.x, durations)  # length equals len(durations)
    indicator_df = pd.DataFrame({
        "stamp": df_reset["stamp"].iloc[1:1+len(fitted_mu)].reset_index(drop=True),
        "bvc": fitted_mu
    })
    indicator_title = "Fitted μ (BSACD1)"

# ---------------------------------------------------------------------------
# Merge indicator with price data (for non-BSACD1, merge on stamp)
# ---------------------------------------------------------------------------
if analysis_type != "BSACD1":
    df_merged = df.merge(indicator_df, on="stamp", how="inner")
    df_merged = df_merged.sort_values("stamp")
    df_merged["bvc"] = df_merged["bvc"].fillna(method="ffill").fillna(0)
else:
    df_merged = indicator_df.copy()

###############################################################################
# 10) PLOTTING THE CHART
###############################################################################
if analysis_type != "BSACD1":
    norm_bvc = plt.Normalize(-1, 1)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    for i in range(len(df_merged)-1):
        xvals = df_merged["stamp"].iloc[i:i+2]
        yvals = df_merged["ScaledPrice"].iloc[i:i+2]
        bvc_val = df_merged["bvc"].iloc[i]
        color = plt.cm.bwr(norm_bvc(bvc_val))
        ax.plot(xvals, yvals, color=color, linewidth=1.2)
    ax.plot(df_merged["stamp"], df_merged["ScaledPrice_EMA"], color="black", linewidth=1, label="EMA(10)")
    ax.plot(df_merged["stamp"], df_merged["vwap_transformed"], color="gray", linewidth=1, label="VWAP")
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Scaled Price", fontsize=8)
    ax.set_title(f"Price with EMA & VWAP (Colored by {indicator_title})", fontsize=10)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.set_ylim(df_merged["ScaledPrice"].min()-50, df_merged["ScaledPrice"].max()+50)
    plt.tight_layout()
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
    ax.plot(df_merged["stamp"], df_merged["bvc"], color="green", linewidth=1.2, label="Fitted μ")
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Fitted μ", fontsize=8)
    ax.set_title("BSACD1 Model: Fitted Conditional Mean Durations", fontsize=10)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    st.pyplot(fig)

if analysis_type != "BSACD1":
    fig_ind, ax_ind = plt.subplots(figsize=(10, 3), dpi=120)
    # For ACD, use the "acd" column instead of "bvc"
    if analysis_type == "ACD":
        ax_ind.plot(indicator_df["stamp"], indicator_df["acd"], color="blue", linewidth=1, label=indicator_title)
        ax_ind.set_ylabel(indicator_title, fontsize=8)
        ax_ind.set_title(f"{indicator_title} Over Time", fontsize=10)
    else:
        ax_ind.plot(indicator_df["stamp"], indicator_df["bvc"], color="blue", linewidth=1, label=indicator_title)
        ax_ind.set_ylabel(indicator_title, fontsize=8)
        ax_ind.set_title(f"{indicator_title} Over Time", fontsize=10)
    ax_ind.set_xlabel("Time", fontsize=8)
    ax_ind.legend(fontsize=7)
    ax_ind.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_ind.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax_ind.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax_ind.get_yticklabels(), fontsize=7)
    st.pyplot(fig_ind)
