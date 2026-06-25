"""Evaluate a model's prediction against reference data and return metrics
plus the reference geometry/overlay for visual comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from backend.geo.chips import load_chip
from backend.models.infer import run_inference
from backend.progress import set_stage

from . import metrics as M
from .reference import osm_features, worldcover_classes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Roads are lines; buffer the reference by ~6 m so it has area to compare against.
ROAD_BUFFER_DEG = 6.0 / 111320.0

# ESA WorldCover 11-class standard palette (value -> name, RGB) for the reference
# overlay rendered on the imagery (left) pane during land-cover evaluation.
WORLDCOVER = {
    10: ("Tree cover", (0, 100, 0)),
    20: ("Shrubland", (255, 187, 34)),
    30: ("Grassland", (255, 255, 76)),
    40: ("Cropland", (240, 150, 255)),
    50: ("Built-up", (250, 0, 0)),
    60: ("Bare / sparse veg", (180, 180, 180)),
    70: ("Snow and ice", (240, 240, 240)),
    80: ("Permanent water", (0, 100, 200)),
    90: ("Herbaceous wetland", (0, 150, 160)),
    95: ("Mangroves", (0, 207, 117)),
    100: ("Moss and lichen", (250, 230, 160)),
}


def _render_worldcover(wc: np.ndarray, chip_id: str, size_px) -> tuple[str, list[dict]]:
    """Colorize the ESA WorldCover reference raster to a georeferenced PNG overlay
    + legend, mirroring the model's land-cover overlay (so the left pane shows the
    reference classes against the model's classes on the right)."""
    h0, w0 = wc.shape
    rgba = np.zeros((h0, w0, 4), dtype=np.uint8)
    counts: dict[int, int] = {}
    for val, (_name, (r, g, b)) in WORLDCOVER.items():
        mask = wc == val
        c = int(mask.sum())
        if c:
            rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = r, g, b
            rgba[mask, 3] = 175
            counts[val] = c
    disp_w, disp_h = (size_px or [w0, h0])
    img = Image.fromarray(rgba, "RGBA").resize((disp_w, disp_h), Image.NEAREST)
    out = DATA_DIR / f"{chip_id}_worldcover.png"
    img.save(out)
    total = sum(counts.values()) or 1
    legend = [
        {"class": WORLDCOVER[v][0], "color": "#%02x%02x%02x" % WORLDCOVER[v][1],
         "pct": round(100.0 * counts[v] / total, 1)}
        for v in sorted(counts, key=lambda k: counts[k], reverse=True)
    ]
    return f"/data/{out.name}", legend


def evaluate(chip_id: str, task: str, model_id: str, prompt: str | None = None) -> dict:
    meta = load_chip(chip_id)
    bounds = meta["bounds"]

    set_stage("eval", "Running model…")
    pred = run_inference(chip_id, task, model_id, prompt)

    if task in ("buildings", "roads"):
        kind = "buildings" if task == "buildings" else "roads"
        set_stage("eval", f"Fetching OSM {kind} (reference)…")
        ref_fc = osm_features(bounds, kind)

        set_stage("eval", "Rasterizing + scoring…")
        w_px, h_px = M.eval_grid(bounds)
        # Roads predictions are already areal SAM polygons; only the line-geometry
        # OSM reference needs buffering to gain comparable area. Buffering the pred
        # too inflates its footprint and unfairly deflates road precision/IoU.
        ref_buf = ROAD_BUFFER_DEG if task == "roads" else 0.0
        pred_mask = M.rasterize_fc(pred["geojson"], bounds, w_px, h_px, buffer_deg=0.0)
        ref_mask = M.rasterize_fc(ref_fc, bounds, w_px, h_px, buffer_deg=ref_buf)
        scores = M.mask_metrics(pred_mask, ref_mask)
        set_stage("done", f"IoU {scores['iou']}")
        return {
            "task": task, "model_id": model_id, "reference": "OpenStreetMap",
            "metrics": scores, "reference_geojson": ref_fc,
            "ref_count": len(ref_fc["features"]),
        }

    if task == "landcover":
        cls_path = DATA_DIR / f"{chip_id}_landcover_cls.npy"
        if not cls_path.exists():
            # Cached result skipped classification; regenerate the class map.
            from backend.models.landcover import classify_landcover

            classify_landcover(chip_id)
        set_stage("eval", "Fetching ESA WorldCover (reference)…")
        wc = worldcover_classes(bounds)
        set_stage("eval", "Comparing classes…")
        agree = M.landcover_agreement(np.load(cls_path), wc, bounds)
        ref_url, ref_legend = _render_worldcover(wc, chip_id, meta.get("size_px"))
        set_stage("done", f"Agreement {agree['overall_agreement']}")
        return {
            "task": task, "model_id": model_id, "reference": "ESA WorldCover",
            "metrics": agree,
            "reference_overlay_url": ref_url,
            "reference_bounds": meta["bounds"],
            "reference_legend": ref_legend,
        }

    raise ValueError(f"No reference evaluation available for task '{task}'.")
