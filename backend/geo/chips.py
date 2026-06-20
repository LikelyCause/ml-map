"""Chip metadata + path helpers shared by ingest and inference."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save_chip(meta: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / f"{meta['id']}.json").write_text(json.dumps(meta))


def load_chip(chip_id: str) -> dict:
    p = DATA_DIR / f"{chip_id}.json"
    if not p.exists():
        raise KeyError(f"Unknown chip '{chip_id}' (fetch imagery first).")
    return json.loads(p.read_text())


def chip_png_path(chip_id: str) -> Path:
    return DATA_DIR / f"{chip_id}_raw.png"
