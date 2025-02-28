import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

###############################################################################
# 1) FETCH DATA
###############################################################################
def fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=1440):
    """
    Fetch OHLCV data from Kraken over 'lookback_minutes' for 'symbol' at 'timeframe'.
    """
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
# 2) DEFINE INDICATOR CLASSES: HawkesBVC, ACDBVC, ACIBVC
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
            current_bvc = current_bvc * alpha_exp + volume.values[i]*labels[i]
            bvc[i] = current_bvc
        if np.max(np.abs(bvc)) != 0:
            bvc = bvc / np.max(np.abs(bvc)) * scale
        result = pd.DataFrame({"stamp": df["stamp"], "bvc": bvc})
        return result

class ACDBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64)//10**9  # seconds
        df["duration"] = df["time_s"].diff().shift(-1)
        df = df.dropna(subset=["duration"])
        df = df[df["duration"]>0]
        if len(df)<10:
            return pd.DataFrame({"stamp":[], "bvc":[]})
        mean_dur = df["duration"].mean()
        std_dur  = df["duration"].std() or 1e-10
        df["std_resid"] = (df["duration"] - mean_dur)/std_dur
        df["price_change"] = np.log(df["close"]/df["close"].shift(1)).fillna(0)
        df["label"] = -df["std_resid"] * df["price_change"]
        df["weighted_volume"] = df["volume"]*df["label"]
        alpha_exp = np.exp(-self.kappa)
        bvc_list=[]
        current_bvc=0.0
        for wv in df["weighted_volume"].values:
            current_bvc = current_bvc*alpha_exp + wv
            bvc_list.append(current_bvc)
        bvc=np.array(bvc_list)
        if np.max(np.abs(bvc))!=0:
            bvc=bvc/np.max(np.abs(bvc))*scale
        df["bvc"]=bvc
        return df[["stamp","bvc"]].copy()

class ACIBVC:
    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def estimate_intensity(self, times, beta):
        intensities=[0.0]
        for i in range(1,len(times)):
            delta_t = times[i]-times[i-1]
            intensities.append(intensities[-1]*np.exp(-beta*delta_t)+1)
        return np.array(intensities)

    def eval(self, df: pd.DataFrame, scale=1e5):
        df = df.copy().sort_values("stamp")
        df["time_s"] = df["stamp"].astype(np.int64)//10**9
        times = df["time_s"].values
        intensities = self.estimate_intensity(times, self.kappa)
        df = df.iloc[:len(intensities)]
        df["intensity"]=intensities
        df["price_change"]=np.log(df["close"]/df["close"].shift(1)).fillna(0)
        df["label"]=df["intensity"]*df["price_change"]
        df["weighted_volume"]=df["volume"]*df["label"]
        alpha_exp=np.exp(-self.kappa)
        bvc_list=[]
        current_bvc=0.0
        for wv in df["weighted_volume"].values:
            current_bvc=current_bvc*alpha_exp+wv
            bvc_list.append(current_bvc)
        bvc=np.array(bvc_list)
        if np.max(np.abs(bvc))!=0:
            bvc=bvc/np.max(np.abs(bvc))*scale
        df["bvc"]=bvc
        return df[["stamp","bvc"]].copy()

###############################################################################
# 3) TUNE KAPPA BY CORRELATION WITH LOG RETURNS
###############################################################################
def tune_kappa_hawkes(df_prices, kappa_grid=None, scale=1e4):
    if kappa_grid is None:
        kappa_grid = [0.01,0.02,0.05,0.1,0.2,0.3,0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa=None
    best_score=-999.0
    for k in kappa_grid:
        model = HawkesBVC(window=20, kappa=k)
        bvc_metrics = model.eval(df_prices, scale=scale)
        merged = df_prices.merge(bvc_metrics,on="stamp", how="inner")
        corr_val = merged[["log_return","bvc"]].corr().iloc[0,1]
        if corr_val>best_score:
            best_score=corr_val
            best_kappa=k
    return best_kappa, best_score

def tune_kappa_acd(df_prices, kappa_grid=None, scale=1e5):
    if kappa_grid is None:
        kappa_grid=[0.01,0.02,0.05,0.1,0.2,0.3,0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"] = np.log(df_prices["close"]).diff()
    best_kappa=None
    best_score=-999.0
    for k in kappa_grid:
        model=ACDBVC(kappa=k)
        bvc_metrics=model.eval(df_prices,scale=scale)
        if len(bvc_metrics)==0:
            continue
        merged=df_prices.merge(bvc_metrics,on="stamp", how="inner")
        corr_val=merged[["log_return","bvc"]].corr().iloc[0,1]
        if corr_val>best_score:
            best_score=corr_val
            best_kappa=k
    return best_kappa,best_score

def tune_kappa_aci(df_prices, kappa_grid=None, scale=1e5):
    if kappa_grid is None:
        kappa_grid=[0.01,0.02,0.05,0.1,0.2,0.3,0.5]
    df_prices = df_prices.copy().sort_values("stamp")
    df_prices["log_return"]=np.log(df_prices["close"]).diff()
    best_kappa=None
    best_score=-999.0
    for k in kappa_grid:
        model=ACIBVC(kappa=k)
        bvc_metrics=model.eval(df_prices,scale=scale)
        if len(bvc_metrics)==0:
            continue
        merged=df_prices.merge(bvc_metrics,on="stamp",how="inner")
        corr_val=merged[["log_return","bvc"]].corr().iloc[0,1]
        if corr_val>best_score:
            best_score=corr_val
            best_kappa=k
    return best_kappa,best_score

###############################################################################
# 4) MAIN SCRIPT (EXAMPLE USAGE)
###############################################################################
if __name__=="__main__":
    # 1) fetch data
    df = fetch_data(symbol="BTC/USD", timeframe="1m", lookback_minutes=720)  # last 12 hours
    df.sort_values("stamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Fetched data range:", df["stamp"].min(), "to", df["stamp"].max())
    print("Number of rows:", len(df))

    # 2) tune hawkes
    best_kappa_hawkes, best_score_hawkes = tune_kappa_hawkes(df, kappa_grid=[0.01,0.05,0.1,0.2,0.3,0.5,0.8,1.0])
    print("Best Hawkes kappa:", best_kappa_hawkes, "with correlation:", best_score_hawkes)

    # 3) tune acd
    best_kappa_acd, best_score_acd = tune_kappa_acd(df, kappa_grid=[0.01,0.05,0.1,0.2,0.5,0.8,1.0])
    print("Best ACD kappa:", best_kappa_acd, "with correlation:", best_score_acd)

    # 4) tune aci
    best_kappa_aci, best_score_aci = tune_kappa_aci(df, kappa_grid=[0.01,0.05,0.1,0.2,0.5,0.8,1.0])
    print("Best ACI kappa:", best_kappa_aci, "with correlation:", best_score_aci)

    # 5) Optionally, plot the best result for, say, Hawkes
    hawkes_best_model = HawkesBVC(window=20, kappa=best_kappa_hawkes)
    hawkes_bvc = hawkes_best_model.eval(df, scale=1e4)
    df_merged = df.merge(hawkes_bvc, on="stamp", how="inner")

    # Plot
    plt.figure(figsize=(10,4))
    plt.title(f"Hawkes BVC with best kappa={best_kappa_hawkes:.3f}")
    plt.plot(df_merged["stamp"], df_merged["close"], label="Price", color="black")
    # Optionally, overlay BVC on a secondary y-axis:
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(df_merged["stamp"], df_merged["bvc"], label="BVC", color="red")
    ax1.set_ylabel("Price")
    ax2.set_ylabel("BVC")
    plt.legend()
    plt.show()
