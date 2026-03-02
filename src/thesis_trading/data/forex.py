from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketDataConfig:
    symbol: str
    start: str
    end: Optional[str]
    interval: str = "1d"


def download_ohlc(config: MarketDataConfig, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        tickers=config.symbol,
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {config.symbol} ({config.interval}).")

    # Normalize columns
    df = df.rename(columns=lambda c: c.strip())
    needed = ["Open", "High", "Low", "Close"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in downloaded data. Got: {list(df.columns)}")

    df = df.reset_index()
    # yfinance sometimes names it 'Date' or 'Datetime'
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Timestamp"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Timestamp"})
    else:
        raise RuntimeError(f"Expected Date/Datetime column. Got: {list(df.columns)}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(None)
    df = df.sort_values("Timestamp").dropna(subset=["Open", "High", "Low", "Close"])

    df.to_csv(out_csv, index=False)
    return out_csv


def load_ohlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df