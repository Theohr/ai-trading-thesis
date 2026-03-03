from __future__ import annotations
import pandas as pd
import numpy as np


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)

    out["log_return"] = np.log(close).diff()
    out["return_1"] = close.pct_change(1)
    out["return_5"] = close.pct_change(5)

    out["vol_10"] = out["return_1"].rolling(10).std()
    out["vol_20"] = out["return_1"].rolling(20).std()

    out["ma_10"] = close.rolling(10).mean()
    out["ma_20"] = close.rolling(20).mean()
    out["ma_ratio"] = out["ma_10"] / out["ma_20"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    return out