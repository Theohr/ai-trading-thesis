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

    # Load + features
    df = load_ohlc(raw_path)
    feat_df = add_basic_features(df)

    # Baseline MA signal (aligned to feat_df index)
    ma_params = cfg["strategies"]["ma_crossover"]
    base_sig = ma_crossover_signals(
        feat_df,
        fast=int(ma_params["fast"]),
        slow=int(ma_params["slow"]),
    )

    # Target: "will this MA signal trade be profitable over next N days?"
    profit_horizon = int(cfg.get("ml_profit_horizon", 3))
    neutral_band = float(cfg.get("ml_neutral_band", 0.0005))
    y = make_signal_profit_target(feat_df, base_sig, horizon=profit_horizon, neutral_band=neutral_band)

    # Feature columns (default: upgraded set)
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

    # Build dataset, but keep Signal too (we'll filter it later)
    data = feat_df[["Timestamp", "Close"] + feature_cols].copy()
    data["Signal"] = base_sig
    data["target"] = y

    # Drop NaNs from rolling features + neutral target zones + last horizon
    data = data.dropna().reset_index(drop=True)

    # Time split
    split_date = pd.to_datetime(cfg.get("ml_split_date", "2019-01-01"))
    train_mask = data["Timestamp"] < split_date
    test_mask = data["Timestamp"] >= split_date

    if train_mask.sum() < 300:
        raise RuntimeError("Not enough training rows before split_date. Adjust split_date or start date.")
    if test_mask.sum() < 200:
        raise RuntimeError("Not enough test rows after split_date. Adjust split_date or end date.")

    X_train = data.loc[train_mask, feature_cols]
    y_train = data.loc[train_mask, "target"].astype(int)

    X_test = data.loc[test_mask, feature_cols]
    y_test = data.loc[test_mask, "target"].astype(int)

    # Train model
    model_name = str(cfg.get("ml_model", "rf")).lower()
    if model_name == "logreg":
        model = train_logistic(X_train, y_train)
    elif model_name in ("rf", "random_forest", "randomforest"):
        model = train_random_forest(X_train, y_train, seed=int(cfg.get("seed", 42)))
    else:
        raise ValueError(f"Unknown ml_model: {model_name}")

    # Predict probability of "profitable trade" (class=1)
    proba_prof = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="proba_profitable")

    # Apply filter on the TEST period only
    thr = float(cfg.get("ml_proba_threshold", 0.6))

    test_data = data.loc[test_mask].reset_index(drop=True)
    test_base_sig = test_data["Signal"].astype(int).reset_index(drop=True)
    test_proba_prof = proba_prof.reset_index(drop=True)

    # Filter rule:
    # - If baseline wants long, require proba_prof >= thr
    # - If baseline wants short, require proba_prof >= thr as well
    # Because proba means "this trade (in that direction) will be profitable"
    # So we keep the sign, just gate it by confidence.
    test_filt_sig = test_base_sig.copy()
    gate = (test_proba_prof >= thr)
    test_filt_sig.loc[~gate] = 0
    test_filt_sig.name = "signal_profit_filtered"

    # Backtest config
    bt_cfg = BacktestConfig(
        initial_cash=float(cfg.get("initial_cash", 10_000)),
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    # Backtest on test period only
    bt_base = backtest_signals(test_data, test_base_sig, bt_cfg)
    bt_filt = backtest_signals(test_data, test_filt_sig, bt_cfg)

    summ_base = performance_summary(bt_base)
    summ_filt = performance_summary(bt_filt)

    out = {
        "experiment": "signal_profit_target",
        "model": model_name,
        "profit_horizon_days": profit_horizon,
        "neutral_band": neutral_band,
        "split_date": str(split_date.date()),
        "threshold": thr,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "baseline_ma_test": summ_base,
        "ma_profit_filtered_test": summ_filt,
        "features": feature_cols,
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/time_split_signal_profit.json").write_text(json.dumps(out, indent=2))
    bt_base.to_csv("reports/bt_time_split_signal_profit_baseline_test.csv", index=False)
    bt_filt.to_csv("reports/bt_time_split_signal_profit_filtered_test.csv", index=False)

    typer.echo("✅ Wrote reports/time_split_signal_profit.json and bt_time_split_signal_profit_*.csv")
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()