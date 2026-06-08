from pathlib import Path
import runpy


if __name__ == "__main__":
    # Legacy implementation bridge
    runpy.run_path(str(Path(__file__).resolve().parent / "02_backtest.py"), run_name="__main__")
