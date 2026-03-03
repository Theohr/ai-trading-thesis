from __future__ import annotations

from pathlib import Path
import json
import typer
import yaml
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from thesis_trading.rl.env import ForexTradingEnv, TradingEnvConfig
from thesis_trading.data.forex import load_ohlc
from thesis_trading.features.technical import add_basic_features

app = typer.Typer(no_args_is_help=True)


def perf_from_equity(equity: pd.Series) -> dict:
    rets = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    ann_ret = float((1.0 + total_return) ** (252.0 / max(len(rets), 1)) - 1.0) if len(rets) > 1 else 0.0
    ann_vol = float(rets.std() * np.sqrt(252.0))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())

    return {
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def position_stats(positions: list[int]) -> dict:
    if not positions:
        return {
            "pct_short": 0.0,
            "pct_flat": 1.0,
            "pct_long": 0.0,
            "num_position_changes": 0,
        }

    pos_arr = np.array(positions, dtype=int)
    n = len(pos_arr)

    pct_short = float(np.mean(pos_arr == -1))
    pct_flat = float(np.mean(pos_arr == 0))
    pct_long = float(np.mean(pos_arr == 1))

    # Count changes between consecutive positions
    num_changes = int(np.sum(pos_arr[1:] != pos_arr[:-1]))

    return {
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "pct_long": pct_long,
        "num_position_changes": num_changes,
    }


@app.command()
def run(config_path: str = typer.Option(..., "--config", "-c")):
    cfg = yaml.safe_load(Path(config_path).read_text())

    symbol = cfg["symbol"]
    interval = cfg.get("interval", "1d")
    raw_path = Path("data/raw") / f"{symbol.replace('=','').replace('/','_')}_{interval}.csv"
    if not raw_path.exists():
        raise RuntimeError(f"Raw data not found: {raw_path}. Run baselines first.")

    split_date = str(cfg.get("rl_split_date", "2019-01-01"))
    seed = int(cfg.get("seed", 42))

    model_path = Path(cfg.get("rl_model_path", "reports/rl/ppo_model.zip"))
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}. Train first.")

    df = load_ohlc(raw_path)
    feat_df = add_basic_features(df)

    feature_cols = cfg.get("rl_feature_cols", None)
    if not feature_cols:
        feature_cols = [
            "return_1", "return_3", "return_5",
            "return_1_lag1", "return_1_lag2", "return_1_lag3", "return_1_lag5",
            "vol_10", "vol_20", "vol_60",
            "vol_percentile_252",
            "ma_ratio_10_20", "ma_ratio_20_50",
            "rsi",
        ]

    data = feat_df[["Timestamp", "Close"] + feature_cols].dropna().reset_index(drop=True)
    split_dt = np.datetime64(split_date)
    test_df = data[data["Timestamp"].values >= split_dt].reset_index(drop=True)

    env_cfg = TradingEnvConfig(
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
        max_episode_steps=None,
        reward_scale=float(cfg.get("rl_reward_scale", 1.0)),
    )
    env = ForexTradingEnv(test_df, feature_cols, env_cfg, seed=seed)

    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    equity = [float(cfg.get("initial_cash", 10_000))]
    positions: list[int] = []
    rewards: list[float] = []
    timestamps: list[pd.Timestamp] = []

    terminated = False
    truncated = False
    t = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        # equity_{t+1} = equity_t * (1 + reward)
        eq_next = equity[-1] * (1.0 + float(reward))
        equity.append(eq_next)

        positions.append(int(info["pos"]))
        rewards.append(float(reward))
        timestamps.append(test_df.loc[min(t, len(test_df) - 1), "Timestamp"])
        t += 1

    eq_series = pd.Series(equity[:-1])  # align with timestamps length
    perf = perf_from_equity(eq_series)
    pos_stats = position_stats(positions)

    out_df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Equity": eq_series.values,
            "Position": positions,
            "Reward": rewards,
        }
    )

    Path("reports/rl").mkdir(parents=True, exist_ok=True)
    out_df.to_csv("reports/rl/eval_equity.csv", index=False)

    summary = {
        "symbol": symbol,
        "interval": interval,
        "split_date": split_date,
        "test_rows": int(len(test_df)),
        "model_path": str(model_path),
        "performance": perf,
        "position_stats": pos_stats,
    }
    Path("reports/rl/eval_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo("✅ Wrote reports/rl/eval_equity.csv and reports/rl/eval_summary.json")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()