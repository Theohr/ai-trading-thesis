# AI Trading Thesis

This repository contains a small research pipeline for comparing:

- rule-based baseline strategies
- supervised ML signal filtering
- time-split ML experiments
- signal-profit classification experiments
- reinforcement learning (PPO) on the same forex dataset

The project currently uses daily EUR/USD data (`EURUSD=X`) downloaded from Yahoo Finance and stores outputs under `reports/`.

## Repository Layout

- `configs/`: YAML configs for each experiment
- `data/raw/`: downloaded market data
- `data/processed/`: cleaned CSV written by baseline runs
- `reports/`: backtests, summaries, threshold sweeps
- `reports/rl/`: RL models and evaluation outputs
- `src/thesis_trading/`: source code and runnable entry points

## Environment Setup

This repo already contains a local virtual environment at `.venv`.

From PowerShell:

```powershell
cd C:\Git\ai-trading-thesis
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To confirm you are using the project environment:

```powershell
Get-Command python
python --version
python -m pip list
```

To leave the environment:

```powershell
deactivate
```

If you need to recreate the environment from scratch:

```powershell
cd C:\Git\ai-trading-thesis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pandas numpy PyYAML typer yfinance scikit-learn gymnasium stable-baselines3 shimmy torch
```

Note: `pyproject.toml` currently does not declare the runtime dependencies, so the explicit `pip install` line above is required on a fresh environment.

## How To Run

All entry points are module-based. Run them from the repository root:

```powershell
python -m thesis_trading.<module_name> -c <config_path>
```

Example:

```powershell
python -m thesis_trading.run_baselines -c configs/forex_daily.yaml
```

Important: most experiments require the raw market data file to already exist. The baseline run is the step that downloads it, so run that first.

## Recommended Run Order

### 1. Baseline Strategies

Downloads OHLC data, writes cleaned CSVs, runs MA crossover and RSI mean reversion backtests.

```powershell
python -m thesis_trading.run_baselines -c configs/forex_daily.yaml
```

Main outputs:

- `data/raw/EURUSDX_1d.csv`
- `data/processed/EURUSDX_1d.csv`
- `reports/bt_ma.csv`
- `reports/bt_rsi.csv`
- `reports/summary.json`

Alternative baseline configs:

```powershell
python -m thesis_trading.run_baselines -c configs/forex_daily_ma_10_30.yaml
python -m thesis_trading.run_baselines -c configs/forex_daily_ma_50_200.yaml
```

### 2. Walk-Forward ML Filter

Builds technical features, trains a walk-forward classifier, and filters MA crossover signals by predicted probability.

```powershell
python -m thesis_trading.run_ml_filter -c configs/forex_daily.yaml
```

Outputs:

- `reports/bt_ma_baseline_aligned.csv`
- `reports/bt_ma_ml_filtered.csv`
- `reports/ml_walkforward_summary.json`

### 3. Walk-Forward Threshold Sweep

Runs one walk-forward model and sweeps multiple probability thresholds.

```powershell
python -m thesis_trading.run_threshold_sweep -c configs/forex_daily.yaml
```

Outputs depend on `ml_model` in config:

- `reports/threshold_sweep_rf.csv`
- `reports/threshold_sweep_rf.json`

or

- `reports/threshold_sweep_logreg.csv`
- `reports/threshold_sweep_logreg.json`

### 4. Time-Split ML Experiment

Uses a fixed train/test date split (`ml_split_date`) instead of walk-forward evaluation.

```powershell
python -m thesis_trading.run_time_split_experiment -c configs/forex_daily.yaml
```

Outputs:

- `reports/time_split_experiment.json`
- `reports/bt_time_split_ma_baseline_test.csv`
- `reports/bt_time_split_ma_ml_filtered_test.csv`

### 5. Signal-Profit Time-Split Experiment

Predicts whether a baseline MA trade will be profitable over the next `N` days, then gates trades using that probability.

```powershell
python -m thesis_trading.run_time_split_signal_profit -c configs/forex_daily.yaml
```

Outputs:

- `reports/time_split_signal_profit.json`
- `reports/bt_time_split_signal_profit_baseline_test.csv`
- `reports/bt_time_split_signal_profit_filtered_test.csv`

### 6. Signal-Profit Threshold Sweep

Sweeps thresholds for the signal-profit classifier.

```powershell
python -m thesis_trading.run_time_split_signal_profit_sweep -c configs/forex_daily_signal_profit_sweep.yaml
```

Outputs:

- `reports/signal_profit_threshold_sweep_rf.csv`
- `reports/signal_profit_threshold_sweep_rf.json`

### 7. RL Train

Trains a PPO agent on the training portion of the dataset defined by `rl_split_date`.

Zero-cost config:

```powershell
python -m thesis_trading.run_rl_train -c configs/forex_daily_rl_cost0.yaml
```

Higher-cost config:

```powershell
python -m thesis_trading.run_rl_train -c configs/forex_daily_rl_cost3.yaml
```

Outputs:

- `reports/rl/train_summary.json`
- `reports/rl/ppo_model_cost0.zip`
- `reports/rl/ppo_model_cost3.zip`

### 8. RL Eval

Evaluates a trained PPO model on the test portion of the dataset.

For cost 0:

```powershell
python -m thesis_trading.run_rl_eval -c configs/forex_daily_rl_cost0.yaml
```

For cost 3:

```powershell
python -m thesis_trading.run_rl_eval -c configs/forex_daily_rl_cost3.yaml
```

Outputs:

- `reports/rl/eval_equity.csv`
- `reports/rl/eval_summary.json`

## Configs

Use these config files depending on what you want to test:

- `configs/forex_daily.yaml`: main all-purpose config
- `configs/forex_daily_ma_10_30.yaml`: MA crossover with 10/30 windows
- `configs/forex_daily_ma_50_200.yaml`: MA crossover with 50/200 windows
- `configs/forex_daily_signal_profit_sweep.yaml`: signal-profit threshold sweep
- `configs/forex_daily_rl_cost0.yaml`: PPO training/eval with zero transaction cost
- `configs/forex_daily_rl_cost3.yaml`: PPO training/eval with 3 bps transaction cost

Key fields to edit:

- `symbol`, `start`, `end`, `interval`
- `cost_bps`, `slippage_bps`, `initial_cash`
- `ml_model`, `ml_proba_threshold`, `ml_thresholds`
- `ml_split_date`, `ml_horizon`, `ml_profit_horizon`
- `rl_split_date`, `rl_train_timesteps`, `rl_model_path`
- `strategies.ma_crossover.fast`
- `strategies.ma_crossover.slow`

## What To Check After Each Run

### Basic sanity checks

- command finishes without a Python traceback
- expected files are written under `data/` or `reports/`
- JSON summaries contain realistic metrics, not all zeros or obviously broken values
- the split-based experiments show enough train and test rows

### Useful checks

Inspect generated reports:

```powershell
Get-Content reports\summary.json
Get-Content reports\ml_walkforward_summary.json
Get-Content reports\time_split_experiment.json
Get-Content reports\time_split_signal_profit.json
Get-Content reports\rl\train_summary.json
Get-Content reports\rl\eval_summary.json
```

Preview CSV outputs:

```powershell
Import-Csv reports\bt_ma.csv | Select-Object -First 5
Import-Csv reports\bt_ma_ml_filtered.csv | Select-Object -First 5
Import-Csv reports\rl\eval_equity.csv | Select-Object -First 5
```

Check that the raw data exists before ML or RL runs:

```powershell
Test-Path data\raw\EURUSDX_1d.csv
```

### Common failure points

- `Raw data not found`: run `run_baselines` first
- `Model not found`: run `run_rl_train` before `run_rl_eval`
- `Not enough training rows` or `Not enough test rows`: move the split date or widen the data range
- missing imports in a fresh environment: install the explicit dependencies listed above

## Quick Start

If you just want the standard end-to-end flow:

```powershell
cd C:\Git\ai-trading-thesis
.\.venv\Scripts\Activate.ps1
python -m thesis_trading.run_baselines -c configs/forex_daily.yaml
python -m thesis_trading.run_ml_filter -c configs/forex_daily.yaml
python -m thesis_trading.run_threshold_sweep -c configs/forex_daily.yaml
python -m thesis_trading.run_time_split_experiment -c configs/forex_daily.yaml
python -m thesis_trading.run_time_split_signal_profit -c configs/forex_daily.yaml
python -m thesis_trading.run_time_split_signal_profit_sweep -c configs/forex_daily_signal_profit_sweep.yaml
python -m thesis_trading.run_rl_train -c configs/forex_daily_rl_cost0.yaml
python -m thesis_trading.run_rl_eval -c configs/forex_daily_rl_cost0.yaml
```

## Notes

- The project uses Yahoo Finance via `yfinance` for market data download.
- The current dataset and reports in the repo are for daily EUR/USD (`EURUSD=X`).
- The RL scripts save models under `reports/rl/` and will overwrite the same path unless you change `rl_model_path`.
