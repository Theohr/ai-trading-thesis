from __future__ import annotations

import numpy as np
import pandas as pd


def make_direction_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Binary target:
      1 if future return over horizon > 0
      0 otherwise
    No leakage: uses shift(-horizon).
    """
    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    y = np.where(future_ret > 0, 1, 0)
    return pd.Series(y, index=df.index, name="target")


def make_direction_target_thresholded(
    df: pd.DataFrame,
    horizon: int = 3,
    neutral_band: float = 0.0005,
) -> pd.Series:
    """
    Binary target with a neutral zone removed:
      1 if future return > +neutral_band
      0 if future return < -neutral_band
      NaN otherwise (neutral/noise zone)

    This can improve signal/noise by ignoring tiny moves.
    """
    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    y = np.where(
        future_ret > neutral_band,
        1,
        np.where(future_ret < -neutral_band, 0, np.nan),
    )
    return pd.Series(y, index=df.index, name="target")


def make_signal_profit_target(
    df: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 3,
    neutral_band: float = 0.0005,
) -> pd.Series:
    """
    Target aligned to the TRADING OBJECTIVE rather than raw direction.

    We assume the signal at time t is used to enter at t+1 (next bar),
    and we evaluate the profit over the next 'horizon' bars from entry.

    entry_return(t) = Close[t+1+h] / Close[t+1] - 1
    profit(t) = signal[t] * entry_return(t)

    Labels:
      1 if profit(t) > +neutral_band
      0 if profit(t) < -neutral_band
      NaN otherwise OR if signal[t] == 0 (no trade)
    """
    close = df["Close"].astype(float)
    s = signal.fillna(0).astype(int).clip(-1, 0, 1) if hasattr(int, "clip") else signal.fillna(0).astype(int)

    # safer clip for pandas series
    s = signal.fillna(0).astype(int).clip(-1, 1)

    entry_px = close.shift(-1)
    exit_px = close.shift(-(1 + horizon))
    entry_ret = (exit_px / entry_px) - 1.0

    profit = s * entry_ret

    y = np.where(
        (s != 0) & (profit > neutral_band),
        1,
        np.where((s != 0) & (profit < -neutral_band), 0, np.nan),
    )

    return pd.Series(y, index=df.index, name="target")