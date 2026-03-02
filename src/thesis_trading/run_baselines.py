from __future__ import annotations

from pathlib import Path
import json
import typer
import yaml

from thesis_trading.data.forex import MarketDataConfig, download_ohlc, load_ohlc
from thesis_trading.strategies.baselines import ma_crossover_signals, rsi_mean_reversion_signals
from thesis_trading.backtest.engine import BacktestConfig, backtest_signals, performance_summary

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(config_path: str = typer.Option(..., "--config", "-c")):
    cfg = yaml.safe_load(Path(config_path).read_text())

    symbol = cfg["symbol"]
    interval = cfg.get("interval", "1d")
    start = cfg["start"]
    end = cfg.get("end")

    raw_path = Path("data/raw") / f"{symbol.replace('=','').replace('/','_')}_{interval}.csv"
    processed_path = Path("data/processed") / f"{symbol.replace('=','').replace('/','_')}_{interval}.csv"

    download_ohlc(MarketDataConfig(symbol=symbol, start=start, end=end, interval=interval), raw_path)
    df = load_ohlc(raw_path)
    df.to_csv(processed_path, index=False)

    bt_cfg = BacktestConfig(
        initial_cash=float(cfg.get("initial_cash", 10_000)),
        cost_bps=float(cfg.get("cost_bps", 1.5)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    # MA crossover
    ma_params = cfg["strategies"]["ma_crossover"]
    sig_ma = ma_crossover_signals(df, fast=int(ma_params["fast"]), slow=int(ma_params["slow"]))
    bt_ma = backtest_signals(df, sig_ma, bt_cfg)
    summ_ma = performance_summary(bt_ma)

    # RSI mean reversion
    rsi_params = cfg["strategies"]["rsi_mean_reversion"]
    sig_rsi = rsi_mean_reversion_signals(
        df,
        period=int(rsi_params["period"]),
        buy_below=float(rsi_params["buy_below"]),
        sell_above=float(rsi_params["sell_above"]),
    )
    bt_rsi = backtest_signals(df, sig_rsi, bt_cfg)
    summ_rsi = performance_summary(bt_rsi)

    Path("reports").mkdir(exist_ok=True)

    bt_ma.to_csv("reports/bt_ma.csv", index=False)
    bt_rsi.to_csv("reports/bt_rsi.csv", index=False)

    summary = {"ma_crossover": summ_ma, "rsi_mean_reversion": summ_rsi}
    Path("reports/summary.json").write_text(json.dumps(summary, indent=2))

    typer.echo("✅ Done. Wrote reports/bt_ma.csv, reports/bt_rsi.csv, reports/summary.json")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()