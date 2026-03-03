from __future__ import annotations

from pathlib import Path
import json
import typer
import yaml
import pandas as pd

from thesis_trading.data.forex import load_ohlc
from thesis_trading.features.technical import add_basic_features
from thesis_trading.models.targets import make_signal_profit_target
from thesis_trading.models.logreg import train_logistic
from thesis_trading.models.rf import train_random_forest
from thesis_trading.strategies.baselines import ma_crossover_signals
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

    # Baseline MA signal
    ma_params = cfg["strategies"]["ma_crossover"]
    base_sig = ma_crossover_signals(
        feat_df,
        fast=int(ma_params["fast"]),
        slow=int(ma_params["slow"]),
    )

    # Profit-target labels
    profit_horizon = int(cfg.get("ml_profit_horizon", 3))
    neutral_band = float(cfg.get("ml_neutral_band", 0.0005))
    y = make_signal_profit_target(feat_df, base_sig, horizon=profit_horizon, neutral_band=neutral_band)

    # Features
    feature_cols = cfg.get("ml_feature_cols", None)
    if not feature_cols:
        feature_cols = [
            "return_1", "return_3", "return_5",
            "return_1_lag1", "return_1_lag2", "return_1_lag3", "return_1_lag5",
            "vol_10", "vol_20", "vol_60",
            "vol_percentile_252",
            "ma_ratio_10_20", "ma_ratio_20_50",
            "rsi",
        ]

    data = feat_df[["Timestamp", "Close"] + feature_cols].copy()
    data["Signal"] = base_sig
    data["target"] = y
    data = data.dropna().reset_index(drop=True)

    # Time split
    split_date = pd.to_datetime(cfg.get("ml_split_date", "2019-01-01"))
    train_mask = data["Timestamp"] < split_date
    test_mask = data["Timestamp"] >= split_date

    if train_mask.sum() < 300:
        raise RuntimeError("Not enough training rows before split_date.")
    if test_mask.sum() < 200:
        raise RuntimeError("Not enough test rows after split_date.")

    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, "target"].astype(int)
    X_test = data.loc[test_mask, feature_cols]

    # Model
    model_name = str(cfg.get("ml_model", "rf")).lower()
    if model_name == "logreg":
        model = train_logistic(X_train, y_train)
    elif model_name in ("rf", "random_forest", "randomforest"):
        model = train_random_forest(X_train, y_train, seed=int(cfg.get("seed", 42)))
    else:
        raise ValueError(f"Unknown ml_model: {model_name}")

    proba_prof = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="proba_profitable")

    # Backtest config
    bt_cfg = BacktestConfig(
        initial_cash=float(cfg.get("initial_cash", 10_000)),
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    # Build test frame
    test_data = data.loc[test_mask].reset_index(drop=True)
    test_base_sig = test_data["Signal"].astype(int).reset_index(drop=True)
    test_proba_prof = proba_prof.reset_index(drop=True)

    # Baseline perf (test only)
    bt_base = backtest_signals(test_data, test_base_sig, bt_cfg)
    base_perf = performance_summary(bt_base)

    thresholds = cfg.get("ml_thresholds", [0.52, 0.55, 0.57, 0.60, 0.62, 0.65])

    rows = []
    for thr in thresholds:
        thr = float(thr)

        # Gate trades by "probability of profitable trade"
        filt_sig = test_base_sig.copy()
        filt_sig.loc[test_proba_prof < thr] = 0
        filt_sig.name = "signal_profit_filtered"

        bt_filt = backtest_signals(test_data, filt_sig, bt_cfg)
        perf = performance_summary(bt_filt)

        rows.append(
            {
                "model": model_name,
                "profit_horizon_days": profit_horizon,
                "neutral_band": neutral_band,
                "split_date": str(split_date.date()),
                "threshold": thr,
                "total_return": perf["total_return"],
                "annualized_return": perf["annualized_return"],
                "annualized_volatility": perf["annualized_volatility"],
                "sharpe": perf["sharpe"],
                "max_drawdown": perf["max_drawdown"],
                "num_position_changes": perf["num_position_changes"],
            }
        )

    out_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    Path("reports").mkdir(exist_ok=True)
    out_csv = f"reports/signal_profit_threshold_sweep_{model_name}.csv"
    out_json = f"reports/signal_profit_threshold_sweep_{model_name}.json"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "experiment": "signal_profit_target_threshold_sweep",
        "model": model_name,
        "profit_horizon_days": profit_horizon,
        "neutral_band": neutral_band,
        "split_date": str(split_date.date()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "baseline_ma_test": base_perf,
        "thresholds": thresholds,
        "features": feature_cols,
        "output_csv": out_csv,
    }
    Path(out_json).write_text(json.dumps(summary, indent=2))

    typer.echo(f"✅ Wrote {out_csv} and {out_json}")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()