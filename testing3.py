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
# 1) FETCH DATA FUNCTIONS
###############################################################################
def fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=1440):
    exchange = ccxt.kraken()
    now_ms = exchange.milliseconds()
    cutoff_ts = now_ms - lookback_minutes * 60 * 1000
    all_ohlcv = []
    since = cutoff_ts
    max_limit = 1440  # max candles per request
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        last_timestamp = ohlcv[-1][0]
        if last_timestamp <= cutoff_ts or len(ohlcv) < max_limit:
            break
        since = last_timestamp + 1
    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["stamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

###############################################################################
# 2) HELPER FUNCTIONS (Indicators, EMA, etc.)
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
def directional_change(prices: NDArray, threshold: float = 0.005) -> NDArray:
    n = len(prices)
    adc = np.zeros(n, dtype=np.float64)
    if n < 2:
        return adc
    ref_price = prices[0]
    direction = 0
    cumulative_change = 0.0
    for i in range(1, n):
        pct_change = (prices[i] - ref_price) / ref_price
        if abs(pct_change) >= threshold:
            if direction == 0 or (direction == 1 and pct_change < -threshold) or (direction == -1 and pct_change > threshold):
                direction = 1 if pct_change > 0 else -1
                ref_price = prices[i]
                cumulative_change = direction * threshold
            else:
                cumulative_change += pct_change - (direction * threshold)
        adc[i] = cumulative_change
    return adc

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
# 3) BS DISTRIBUTION FUNCTIONS (for BSACD1 model)
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
# 4) BSACD1 MODEL (Mean-based) IMPLEMENTATION
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
# 5) INDICATOR CLASSES
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
        df["time_s"] = df["stamp"].astype(np.int64)//10**9
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
        df["time_s"] = df["stamp"].astype(np.int64)//10**9
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
# 6) TUNING FUNCTIONS (using correlation with log returns)
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

def tune_kappa_acd(df_prices, kappa_grid=None, scale=1e5):
    if kappa_grid is None:
        kappa_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa = None
    best_score = -999.0
    for k in kappa_grid:
        model = ACDBVC(kappa=k)
        bvc_metrics = model.eval(df_prices, scale=scale)
        if len(bvc_metrics) == 0:
            continue
        merged = df_prices.merge(bvc_metrics, on="stamp", how="inner")
        corr_val = merged[["log_return", "bvc"]].corr().iloc[0, 1]
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
# 7) MAIN SCRIPT (STREAMLIT APP)
###############################################################################
st.header("Price & Indicator Analysis")

# Fetch data (e.g., last 12 hours)
df = fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=720)
df = df.sort_values("stamp").reset_index(drop=True)
st.write("Data range:", df["stamp"].min(), "to", df["stamp"].max())
st.write("Number of rows:", len(df))

# Tune parameters for each indicator and display results
best_kappa_hawkes, best_score_hawkes = tune_kappa_hawkes(df, kappa_grid=[0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.0])
st.write("Best Hawkes kappa:", best_kappa_hawkes, "with correlation:", best_score_hawkes)

best_kappa_acd, best_score_acd = tune_kappa_acd(df, kappa_grid=[0.01,0.05,0.1,0.2,0.5,0.8,1.0])
st.write("Best ACD kappa:", best_kappa_acd, "with correlation:", best_score_acd)

best_kappa_aci, best_score_aci = tune_kappa_aci(df, kappa_grid=[0.01,0.05,0.1,0.2,0.5,0.8,1.0])
st.write("Best ACI kappa:", best_kappa_aci, "with correlation:", best_score_aci)

# ---------------------------------------------------------------------------
# Indicator Evaluation: use selected indicator based on analysis_type
# ---------------------------------------------------------------------------
if analysis_type == "Hawkes BVC":
    st.write("### Hawkes BVC Analysis")
    indicator_title = "BVC"
    indicator_df = HawkesBVC(window=20, kappa=best_kappa_hawkes).eval(df, scale=1e4)
elif analysis_type == "ADC":
    st.write("### Average Directional Change (ADC) Analysis")
    threshold_param = st.slider("Directional Change Threshold (%)", min_value=0.1, max_value=5.0,
                                  value=0.5, step=0.1) / 100
    indicator_title = "ADC"
    # Use the same directional_change function defined earlier
    # Here we simply compute the ADC indicator without smoothing for demonstration.
    df_temp = df.copy().sort_values("stamp")
    adc_vals = directional_change(df_temp["close"].values, threshold=threshold_param)
    # Scale ADC indicator
    if np.max(np.abs(adc_vals)) != 0:
        adc_vals = adc_vals / np.max(np.abs(adc_vals)) * 1e4
    indicator_df = pd.DataFrame({"stamp": df_temp["stamp"], "bvc": adc_vals})
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
    durations = df_reset["duration"].dropna().values  # length = n-1
    init_params = [0.0, 0.5, 0.0, 1.0]
    res = minimize(bsacd1_negloglik, init_params, args=(durations,), method="L-BFGS-B")
    st.write("Estimated BSACD1 parameters:", res.x)
    fitted_mu = compute_fitted_mu(res.x, durations)
    # Use the fitted μ starting from the second observation
    indicator_df = pd.DataFrame({
        "stamp": df_reset["stamp"].iloc[1:].reset_index(drop=True),
        "bvc": fitted_mu[1:]
    })
    indicator_title = "Fitted μ"

# ---------------------------------------------------------------------------
# Merge indicator with price data for plotting
# ---------------------------------------------------------------------------
if analysis_type != "BSACD1":
    df_merged = df.merge(indicator_df, on="stamp", how="inner")
    df_merged = df_merged.sort_values("stamp")
    df_merged["bvc"] = df_merged["bvc"].fillna(method="ffill").fillna(0)
else:
    df_merged = indicator_df.copy()

###############################################################################
# 8) PLOTTING THE CHART
###############################################################################
# For non-BSACD1 analyses, plot a gradient-colored price chart.
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
    # For BSACD1, plot the fitted μ (conditional mean durations)
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

# Optionally, plot the raw indicator (for non-BSACD1)
if analysis_type != "BSACD1":
    fig_ind, ax_ind = plt.subplots(figsize=(10, 3), dpi=120)
    ax_ind.plot(indicator_df["stamp"], indicator_df["bvc"], color="blue", linewidth=1, label=indicator_title)
    ax_ind.set_xlabel("Time", fontsize=8)
    ax_ind.set_ylabel(indicator_title, fontsize=8)
    ax_ind.legend(fontsize=7)
    ax_ind.set_title(f"{indicator_title} Over Time", fontsize=10)
    ax_ind.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_ind.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.setp(ax_ind.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax_ind.get_yticklabels(), fontsize=7)
    st.pyplot(fig_ind)
