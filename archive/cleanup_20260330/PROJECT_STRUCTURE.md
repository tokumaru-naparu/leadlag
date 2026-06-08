# Leadlag Project Structure

## Standard layout

- `data/raw/`: downloaded source data (ETF prices, returns)
- `data/processed/`: canonical processed datasets (signals/trades/returns/market/performance/analysis)
- `data/external/`: market environment data (vix/usdjpy/sp500)
- `scripts/`: executable strategy and analysis scripts
- `output/charts/`: generated charts
- `output/reports/`: text/csv reports
- `output/results/`: numeric result csv files
- `config.py`: centralized path constants

## Script entry points and role

- `scripts/01_fetch_data.py`: download US/JP ETF raw data
- `scripts/02_preprocess.py`: preprocess pipeline (bridge to legacy `02_backtest.py`)
- `scripts/03_backtest_baseline.py`: baseline backtest (bridge to legacy `03_result.py`)
- `scripts/04_backtest_hplus.py`: H+ backtest (bridge to legacy `20_filter_backtest.py`)
- `scripts/05_sizing_comparison.py`: sizing-style comparison (bridge to legacy `06_backtest_double.py`)
- `scripts/06_filter_backtest.py`: filter backtest (bridge to legacy `20_filter_backtest.py`)
- `scripts/07_grid_search.py`: H+ filter grid search (bridge to legacy `scripts/data/21.py`)
- `scripts/08_signal_today.py`: daily signal operation
- `scripts/utils.py`: shared path utilities

## Legacy scripts kept for backward compatibility

- Existing numbered scripts (`02_backtest.py`, `03_result.py`, `04_backtest_pca_plain.py`, `05.py`, `06_backtest_double.py`, ... `20_filter_backtest.py`) are preserved.
- New entry points call legacy implementations so current workflows continue to run.
