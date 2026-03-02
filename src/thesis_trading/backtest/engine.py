from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    cost_bps: float = 1.5
    slippage_bps: float = 0.0


def backtest_signals(df: pd.DataFrame, signal: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Simple bar-by-bar backtest:
    - signal in {-1,0,1} indicates desired position for NEXT bar (shifted to avoid lookahead)
    - position is held until signal changes
    - PnL computed from Close-to-Close returns
    - costs applied on position changes (turnover)
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain 'Close'")

    px = df["Close"].astype(float)
    ret = px.pct_change().fillna(0.0)

    desired = signal.fillna(0).astype(int).clip(-1, 1)
    pos = desired.shift(1).fillna(0).astype(int)  # avoid lookahead

    # Turnover: when position changes
    pos_change = (pos - pos.shift(1).fillna(0)).abs()

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10_000.0
    costs = pos_change * cost_rate  # proportional cost

    strat_ret = pos * ret - costs

    equity = (1.0 + strat_ret).cumprod() * cfg.initial_cash

    out = pd.DataFrame(
        {
            "Timestamp": df["Timestamp"],
            "Close": px,
            "Return": ret,
            "Signal": desired,
            "Position": pos,
            "Turnover": pos_change,
            "Costs": costs,
            "StrategyReturn": strat_ret,
            "Equity": equity,
        }
    )
    return out


def performance_summary(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    r = bt["StrategyReturn"].astype(float)
    if r.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = (r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year)

    equity = bt["Equity"].astype(float)
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = dd.min()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    ann_return = (1 + total_return) ** (periods_per_year / max(1, len(r))) - 1.0
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)

    # Trade-ish stats (approx): count position flips
    trades = int((bt["Turnover"] > 0).sum())

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "num_position_changes": trades,
    }