from pathlib import Path
import runpy


if __name__ == "__main__":
    # Legacy implementation bridge
    runpy.run_path(str(Path(__file__).resolve().parent / "20_filter_backtest.py"), run_name="__main__")
