from __future__ import annotations
import pandas as pd
import numpy as np


def make_direction_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Predict whether future return over horizon is positive or negative.
    No leakage: future shift.
    """
    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    y = np.where(future_ret > 0, 1, 0)
    return pd.Series(y, index=df.index, name="target")