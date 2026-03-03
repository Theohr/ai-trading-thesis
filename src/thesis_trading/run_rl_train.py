from __future__ import annotations

from pathlib import Path
import json
import typer
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from thesis_trading.rl.env import ForexTradingEnv, TradingEnvConfig
from thesis_trading.data.forex import load_ohlc
from thesis_trading.features.technical import add_basic_features

app = typer.Typer(no_args_is_help=True)


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

    # Allow config to control save path (supports both with/without .zip)
    model_path_cfg = str(cfg.get("rl_model_path", "reports/rl/ppo_model.zip"))
    model_path = Path(model_path_cfg)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_stem = str(model_path.with_suffix(""))  # SB3 adds .zip automatically if missing

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

    train_df = data[data["Timestamp"].values < split_dt].reset_index(drop=True)
    if len(train_df) < 800:
        raise RuntimeError(f"Not enough training rows: {len(train_df)}. Adjust rl_split_date or start date.")

    env_cfg = TradingEnvConfig(
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
        max_episode_steps=int(cfg.get("rl_max_episode_steps", 0)) or None,
        reward_scale=float(cfg.get("rl_reward_scale", 1.0)),
    )

    def make_env():
        return ForexTradingEnv(train_df, feature_cols, env_cfg, seed=seed)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=float(cfg.get("rl_lr", 3e-4)),
        n_steps=int(cfg.get("rl_n_steps", 2048)),
        batch_size=int(cfg.get("rl_batch_size", 64)),
        gamma=float(cfg.get("rl_gamma", 0.99)),
        gae_lambda=float(cfg.get("rl_gae_lambda", 0.95)),
        ent_coef=float(cfg.get("rl_ent_coef", 0.0)),
    )

    timesteps = int(cfg.get("rl_train_timesteps", 300_000))

    model.learn(total_timesteps=timesteps)
    model.save(model_stem)

    actual_saved = str(Path(model_stem).with_suffix(".zip"))

    out = {
        "symbol": symbol,
        "interval": interval,
        "split_date": split_date,
        "train_rows": int(len(train_df)),
        "feature_cols": feature_cols,
        "env": {
            "cost_bps": env_cfg.cost_bps,
            "slippage_bps": env_cfg.slippage_bps,
            "max_episode_steps": env_cfg.max_episode_steps,
            "reward_scale": env_cfg.reward_scale,
        },
        "ppo": {
            "timesteps": timesteps,
            "lr": float(cfg.get("rl_lr", 3e-4)),
            "n_steps": int(cfg.get("rl_n_steps", 2048)),
            "batch_size": int(cfg.get("rl_batch_size", 64)),
            "gamma": float(cfg.get("rl_gamma", 0.99)),
            "gae_lambda": float(cfg.get("rl_gae_lambda", 0.95)),
            "ent_coef": float(cfg.get("rl_ent_coef", 0.0)),
        },
        "model_path": actual_saved,
    }

    Path("reports/rl").mkdir(parents=True, exist_ok=True)
    Path("reports/rl/train_summary.json").write_text(json.dumps(out, indent=2))
    typer.echo("✅ Trained PPO. Wrote reports/rl/train_summary.json and model.")
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()