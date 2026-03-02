from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    close = df["Close"].astype(float)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    # Signal: +1 when fast>slow, -1 when fast<slow
    sig = np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0))
    return pd.Series(sig, index=df.index, name="signal_ma")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    # Wilder's smoothing approximation using EMA
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    period: int,
    buy_below: float,
    sell_above: float,
) -> pd.Series:
    close = df["Close"].astype(float)
    r = rsi(close, period=period)

    # Mean reversion:
    # RSI < buy_below => long
    # RSI > sell_above => short
    sig = np.where(r < buy_below, 1, np.where(r > sell_above, -1, 0))
    return pd.Series(sig, index=df.index, name="signal_rsi")