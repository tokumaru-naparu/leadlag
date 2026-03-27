from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Backward-compatible legacy locations
LEGACY_SCRIPTS_DATA_DIR = ROOT / "scripts" / "data"
LEGACY_HISTORY_DIR = DATA_DIR / "history"
LEGACY_SCRIPTS_HISTORY_DIR = LEGACY_SCRIPTS_DATA_DIR / "history"

OUTPUT_DIR = ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"


def ensure_directories() -> None:
    for p in [
        RAW_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        CHARTS_DIR,
        REPORTS_DIR,
        RESULTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def pick_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def pick_largest_csv(paths: list[Path]) -> Path | None:
    best_path = None
    best_rows = -1
    for p in paths:
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                rows = sum(1 for _ in f) - 1
        except Exception:
            rows = -1
        if rows > best_rows:
            best_rows = rows
            best_path = p
    return best_path


ensure_directories()
