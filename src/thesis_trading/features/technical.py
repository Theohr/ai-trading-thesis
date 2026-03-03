from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set for FX daily prediction / filtering.
    Includes:
    - returns + lagged returns
    - volatility + regime proxy (volatility percentile)
    - moving average ratios
    - RSI
    """
    out = df.copy()
    close = out["Close"].astype(float)

    # Returns
    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["return_5"] = close.pct_change(5)
    out["log_return"] = np.log(close).diff()

    # Lagged returns (common in quant features)
    for lag in [1, 2, 3, 5]:
        out[f"return_1_lag{lag}"] = out["return_1"].shift(lag)

    # Volatility
    out["vol_10"] = out["return_1"].rolling(10).std()
    out["vol_20"] = out["return_1"].rolling(20).std()
    out["vol_60"] = out["return_1"].rolling(60).std()

    # Regime feature: volatility percentile over last ~1 trading year
    # This is a simple regime proxy: "how high is vol relative to recent history"
    vol_rank_window = 252
    out["vol_percentile_252"] = (
        out["vol_20"]
        .rolling(vol_rank_window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # Moving averages + ratios
    out["ma_10"] = close.rolling(10).mean()
    out["ma_20"] = close.rolling(20).mean()
    out["ma_50"] = close.rolling(50).mean()
    out["ma_ratio_10_20"] = out["ma_10"] / out["ma_20"]
    out["ma_ratio_20_50"] = out["ma_20"] / out["ma_50"]

    # RSI (simple rolling version)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    return out