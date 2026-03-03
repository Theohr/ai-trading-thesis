from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd

from thesis_trading.data.forex import load_ohlc
from thesis_trading.features.technical import add_basic_features


@dataclass(frozen=True)
class RLDatasetConfig:
    split_date: str = "2019-01-01"


DEFAULT_FEATURES: List[str] = [
    "return_1", "return_3", "return_5",
    "return_1_lag1", "return_1_lag2", "return_1_lag3", "return_1_lag5",
    "vol_10", "vol_20", "vol_60",
    "vol_percentile_252",
    "ma_ratio_10_20", "ma_ratio_20_50",
    "rsi",
]


def build_rl_datasets(raw_csv_path: str, split_date: str, feature_cols: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = load_ohlc(pd.Path(raw_csv_path) if hasattr(pd, "Path") else raw_csv_path)  # fallback
    feat_df = add_basic_features(df)

    cols = feature_cols or DEFAULT_FEATURES
    data = feat_df[["Timestamp", "Close"] + cols].dropna().reset_index(drop=True)

    split_dt = pd.to_datetime(split_date)
    train = data[data["Timestamp"] < split_dt].reset_index(drop=True)
    test = data[data["Timestamp"] >= split_dt].reset_index(drop=True)

    if len(train) < 500 or len(test) < 300:
        raise RuntimeError(f"Not enough rows after split. train={len(train)} test={len(test)}")

    return train, test, cols