from __future__ import annotations
import pandas as pd


def apply_proba_filter(
    baseline_signal: pd.Series,
    proba_up: pd.Series,
    threshold: float = 0.55,
) -> pd.Series:
    """
    Filter baseline long/short trades using model confidence.
    - If baseline wants LONG (+1): allow only if proba_up >= threshold
    - If baseline wants SHORT (-1): allow only if proba_up <= (1 - threshold)
    - If baseline 0: keep 0
    """
    s = baseline_signal.fillna(0).astype(int).clip(-1, 1)
    p = proba_up

    out = pd.Series(0, index=s.index, dtype=int, name="signal_ml_filtered")

    long_ok = (s == 1) & (p >= threshold)
    short_ok = (s == -1) & (p <= (1.0 - threshold))

    out[long_ok] = 1
    out[short_ok] = -1
    return out