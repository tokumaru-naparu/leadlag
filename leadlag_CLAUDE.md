# Lead-Lag Strategy Design Notes

## Objective
Build a lead-lag model between US sector ETFs and JP sector ETFs, then test whether US moves can predict next-session JP sector returns.

## Files
- `01_fetch_data.py`: Fetch and store input market data
- `02_backtest.py`: Run signal generation and backtest logic
- `03_result.py`: Summarize and visualize backtest output

## Data Inputs
- `data/us_etf.csv`: Universe of US sector ETFs
- `data/jp_etf.csv`: Universe of JP sector ETFs

## Suggested Workflow
1. Fetch prices for all US and JP symbols.
2. Align calendars and compute daily returns.
3. Estimate lagged relationship (correlation/regression).
4. Generate tradable signals for JP sectors.
5. Backtest with transaction costs and simple position sizing.
6. Export metrics and plots.

## Notes
- Keep all timestamps timezone-aware.
- Ensure survivorship assumptions are documented.
- Save intermediate datasets for reproducibility.
