"""Phase 0 demo data.

Generates a synthetic 'satellite-like' RGB chip and a matching 'annotation'
overlay so the split-view UI has something to render before the real
STAC ingest + model inference pipelines (Phase 1+) are wired up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A small AOI over downtown Denver, CO (NAIP-covered, US). [west, south, east, north]
DEMO_BOUNDS = [-104.9975, 39.7440, -104.9890, 39.7505]
SIZE = 512


def _synthetic_imagery(rng: np.random.Generator) -> np.ndarray:
    """Fake aerial RGB: green base, gray 'road' grid, tan 'building' blocks."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # vegetation base with noise
    img[..., 0] = 70 + rng.integers(0, 25, (SIZE, SIZE))
    img[..., 1] = 110 + rng.integers(0, 30, (SIZE, SIZE))
    img[..., 2] = 60 + rng.integers(0, 25, (SIZE, SIZE))

    # road grid
    for x in range(40, SIZE, 110):
        img[:, x : x + 12] = (105, 105, 110)
    for y in range(60, SIZE, 120):
        img[y : y + 12, :] = (105, 105, 110)

    # building blocks
    for _ in range(28):
        y, x = rng.integers(0, SIZE - 45, 2)
        h, w = rng.integers(18, 42, 2)
        shade = rng.integers(150, 200)
        img[y : y + h, x : x + w] = (shade, shade - 20, shade - 40)

    return img


def _annotation(rng: np.random.Generator) -> np.ndarray:
    """RGBA overlay placeholder: red roads + blue building outlines."""
    over = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    for x in range(40, SIZE, 110):
        over[:, x : x + 12] = (255, 70, 70, 150)

    for y in range(60, SIZE, 120):
        over[y : y + 12, :] = (255, 70, 70, 150)

    rng2 = np.random.default_rng(7)

    for _ in range(28):
        y, x = rng2.integers(0, SIZE - 45, 2)
        h, w = rng2.integers(18, 42, 2)
        over[y : y + h, x : x + w] = (60, 130, 255, 130)

    return over


def ensure_demo() -> dict:
    """Create the demo PNGs if missing; return metadata for the API."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "demo_raw.png"
    ann_path = DATA_DIR / "demo_annotated.png"

    if not raw_path.exists() or not ann_path.exists():
        rng = np.random.default_rng(42)
        Image.fromarray(_synthetic_imagery(rng), "RGB").save(raw_path)
        Image.fromarray(_annotation(rng), "RGBA").save(ann_path)

    return {
        "id": "demo",
        "bounds": DEMO_BOUNDS,
        "raw_url": "/data/demo_raw.png",
        "annotated_url": "/data/demo_annotated.png",
        "note": "Phase 0 synthetic demo chip (Denver, CO AOI).",
    }
