from __future__ import annotations

from pathlib import Path
import json
import typer
import yaml

import pandas as pd

from thesis_trading.data.forex import load_ohlc
from thesis_trading.features.technical import add_basic_features
from thesis_trading.models.targets import make_direction_target
from thesis_trading.models.walkforward import WalkForwardConfig, walk_forward_predict_proba
from thesis_trading.strategies.baselines import ma_crossover_signals
from thesis_trading.strategies.ml_filter import apply_proba_filter
from thesis_trading.backtest.engine import BacktestConfig, backtest_signals, performance_summary

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(config_path: str = typer.Option(..., "--config", "-c")):
    cfg = yaml.safe_load(Path(config_path).read_text())

    symbol = cfg["symbol"]
    interval = cfg.get("interval", "1d")

    raw_path = Path("data/raw") / f"{symbol.replace('=','').replace('/','_')}_{interval}.csv"
    if not raw_path.exists():
        raise RuntimeError(f"Raw data not found: {raw_path}. Run baselines first to download data.")

    df = load_ohlc(raw_path)

    # Features + target
    feat_df = add_basic_features(df)

    y = make_direction_target(feat_df, horizon=1)

    feature_cols = [
        "return_1", "return_5",
        "vol_10", "vol_20",
        "ma_ratio",
        "rsi",
    ]

    data = feat_df[["Timestamp", "Close"] + feature_cols].copy()
    data["target"] = y

    # Drop rows with NaNs (rolling features + last horizon rows)
    data = data.dropna().reset_index(drop=True)

    X = data[feature_cols]
    y = data["target"].astype(int)

    # Walk-forward probabilities
    wf_cfg = WalkForwardConfig(
        train_size=int(cfg.get("ml_train_size", 1000)),
        test_size=int(cfg.get("ml_test_size", 250)),
        min_train=int(cfg.get("ml_min_train", 500)),
        proba_threshold=float(cfg.get("ml_proba_threshold", 0.55)),
    )
    proba_up, clf_metrics = walk_forward_predict_proba(X, y, wf_cfg)

    # Baseline MA signals aligned to 'data' index (NOT original df index)
    base_sig = ma_crossover_signals(data, fast=int(cfg["strategies"]["ma_crossover"]["fast"]),
                                    slow=int(cfg["strategies"]["ma_crossover"]["slow"]))

    # ML filtered signals
    filt_sig = apply_proba_filter(base_sig, proba_up, threshold=wf_cfg.proba_threshold)

    # Backtest on same aligned frame
    bt_cfg = BacktestConfig(
        initial_cash=float(cfg.get("initial_cash", 10_000)),
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    bt_base = backtest_signals(data, base_sig, bt_cfg)
    bt_filt = backtest_signals(data, filt_sig, bt_cfg)

    summ_base = performance_summary(bt_base)
    summ_filt = performance_summary(bt_filt)

    Path("reports").mkdir(exist_ok=True)
    bt_base.to_csv("reports/bt_ma_baseline_aligned.csv", index=False)
    bt_filt.to_csv("reports/bt_ma_ml_filtered.csv", index=False)

    summary = {
        "classification": clf_metrics,
        "baseline_ma": summ_base,
        "ma_ml_filtered": summ_filt,
        "ml_config": {
            "train_size": wf_cfg.train_size,
            "test_size": wf_cfg.test_size,
            "min_train": wf_cfg.min_train,
            "proba_threshold": wf_cfg.proba_threshold,
            "features": feature_cols,
            "target": "next-day direction",
        },
    }
    Path("reports/ml_walkforward_summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo("✅ Done. Wrote reports/bt_ma_ml_filtered.csv and reports/ml_walkforward_summary.json")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()