"""Evaluate a model's prediction against reference data and return metrics
plus the reference geometry/overlay for visual comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.geo.chips import load_chip
from backend.models.infer import run_inference
from backend.progress import set_stage

from . import metrics as M
from .reference import osm_features, worldcover_classes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Roads are lines; buffer the reference by ~6 m so it has area to compare against.
ROAD_BUFFER_DEG = 6.0 / 111320.0


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
        set_stage("done", f"Agreement {agree['overall_agreement']}")
        return {
            "task": task, "model_id": model_id, "reference": "ESA WorldCover",
            "metrics": agree,
        }

    raise ValueError(f"No reference evaluation available for task '{task}'.")
