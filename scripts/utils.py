from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def data_paths() -> dict[str, Path]:
    data = ROOT / "data"
    return {
        "root": ROOT,
        "data": data,
        "raw": data / "raw",
        "processed": data / "processed",
        "external": data / "external",
        "legacy_data": ROOT / "scripts" / "data",
        "legacy_history": data / "history",
        "legacy_scripts_history": ROOT / "scripts" / "data" / "history",
        "output": ROOT / "output",
    }


def existing_path(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None
