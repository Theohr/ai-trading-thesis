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
from thesis_trading.models.logreg import train_logistic
from thesis_trading.models.rf import train_random_forest
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
        raise RuntimeError(f"Raw data not found: {raw_path}. Run baselines first.")

    df = load_ohlc(raw_path)
    feat_df = add_basic_features(df)
    y = make_direction_target(feat_df, horizon=1)

    feature_cols = [
        "return_1", "return_5",
        "vol_10", "vol_20",
        "ma_ratio", "rsi",
    ]

    data = feat_df[["Timestamp", "Close"] + feature_cols].copy()
    data["target"] = y
    data = data.dropna().reset_index(drop=True)

    X = data[feature_cols]
    y = data["target"].astype(int)

    # Choose model
    model_name = str(cfg.get("ml_model", "logreg")).lower()
    if model_name == "logreg":
        trainer = train_logistic
    elif model_name in ("rf", "random_forest", "randomforest"):
        seed = int(cfg.get("seed", 42))
        trainer = lambda Xtr, ytr: train_random_forest(Xtr, ytr, seed=seed)
    else:
        raise ValueError(f"Unknown ml_model: {model_name}")

    # Walk-forward proba once (thresholds applied later)
    wf_cfg = WalkForwardConfig(
        train_size=int(cfg.get("ml_train_size", 1000)),
        test_size=int(cfg.get("ml_test_size", 250)),
        min_train=int(cfg.get("ml_min_train", 500)),
        proba_threshold=0.55,  # placeholder, we sweep below
    )
    proba_up, clf_metrics = walk_forward_predict_proba(X, y, wf_cfg, trainer=trainer)

    # Baseline signals aligned
    ma_params = cfg["strategies"]["ma_crossover"]
    base_sig = ma_crossover_signals(data, fast=int(ma_params["fast"]), slow=int(ma_params["slow"]))

    bt_cfg = BacktestConfig(
        initial_cash=float(cfg.get("initial_cash", 10_000)),
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    base_bt = backtest_signals(data, base_sig, bt_cfg)
    base_perf = performance_summary(base_bt)

    thresholds = cfg.get("ml_thresholds", [0.50, 0.52, 0.55, 0.60, 0.65])
    rows = []

    for thr in thresholds:
        filt_sig = apply_proba_filter(base_sig, proba_up, threshold=float(thr))
        bt = backtest_signals(data, filt_sig, bt_cfg)
        perf = performance_summary(bt)
        rows.append(
            {
                "model": model_name,
                "threshold": float(thr),
                "total_return": perf["total_return"],
                "annualized_return": perf["annualized_return"],
                "annualized_volatility": perf["annualized_volatility"],
                "sharpe": perf["sharpe"],
                "max_drawdown": perf["max_drawdown"],
                "num_position_changes": perf["num_position_changes"],
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["model", "threshold"]).reset_index(drop=True)

    Path("reports").mkdir(exist_ok=True)
    out_df.to_csv(f"reports/threshold_sweep_{model_name}.csv", index=False)

    summary = {
        "classification": clf_metrics,
        "baseline_ma": base_perf,
        "model": model_name,
        "thresholds": thresholds,
        "features": feature_cols,
        "target": "next-day direction",
        "output_csv": f"reports/threshold_sweep_{model_name}.csv",
    }
    Path(f"reports/threshold_sweep_{model_name}.json").write_text(json.dumps(summary, indent=2))

    typer.echo(f"✅ Wrote reports/threshold_sweep_{model_name}.csv and .json")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()