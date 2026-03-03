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


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance sometimes returns MultiIndex columns: (price_field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # keep only OHLCV field name
    df = df.rename(columns=lambda c: str(c).strip())
    return df


def download_ohlc(config: MarketDataConfig, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        tickers=config.symbol,
        start=config.start,
        end=config.end,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        group_by="column",  # helps keep OHLC columns more consistent
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {config.symbol} ({config.interval}).")

    df = _flatten_columns(df)

    # Normalize expected OHLC names (case variations happen)
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc == "close":
            col_map[c] = "Close"
        elif lc == "adj close":
            col_map[c] = "Adj Close"
        elif lc == "volume":
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)

    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing OHLC columns {missing}. Got columns: {list(df.columns)}")

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "Timestamp"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Timestamp"})
    else:
        # Sometimes index name comes through as None
        if df.columns[0] not in ("Timestamp",):
            # try to rename first column to Timestamp
            df = df.rename(columns={df.columns[0]: "Timestamp"})

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True).dt.tz_convert(None)

    df = (
        df.sort_values("Timestamp")
        .dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
        .reset_index(drop=True)
    )

    df.to_csv(out_csv, index=False)
    return out_csv


def load_ohlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df