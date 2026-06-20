"""Evaluation metrics: prediction vs reference.

Vector tasks (buildings, roads) -> rasterize both to a common grid and compute
pixel IoU / precision / recall / F1. Land cover -> map both the 13-class crop
prediction and ESA WorldCover to a shared coarse scheme and report agreement.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape


def eval_grid(bounds, target=1024):
    w, s, e, n = bounds
    aspect = (e - w) / (n - s)
    if aspect >= 1:
        return target, max(1, round(target / aspect))
    return max(1, round(target * aspect)), target


def rasterize_fc(fc: dict, bounds, w_px, h_px, buffer_deg: float = 0.0) -> np.ndarray:
    geoms = []
    for f in fc.get("features", []):
        try:
            g = shape(f["geometry"])
        except Exception:  # noqa: BLE001
            continue
        if buffer_deg > 0:
            g = g.buffer(buffer_deg)
        if not g.is_empty:
            geoms.append(g)
    if not geoms:
        return np.zeros((h_px, w_px), dtype=bool)
    tr = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w_px, h_px)
    arr = rasterize(
        [(mapping(g), 1) for g in geoms],
        out_shape=(h_px, w_px), transform=tr, fill=0, dtype="uint8", all_touched=True,
    )
    return arr.astype(bool)


def mask_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    inter = int((pred & ref).sum())
    union = int((pred | ref).sum())
    p_sum, r_sum = int(pred.sum()), int(ref.sum())
    iou = inter / union if union else 0.0
    precision = inter / p_sum if p_sum else 0.0
    recall = inter / r_sum if r_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "iou": round(iou, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pred_coverage_pct": round(100 * p_sum / pred.size, 2),
        "ref_coverage_pct": round(100 * r_sum / ref.size, 2),
    }


# Coarse land-cover scheme shared by the crop model and ESA WorldCover.
COARSE = ["cropland", "tree", "veg", "wetland", "water", "built", "other"]
_C = {name: i for i, name in enumerate(COARSE)}

# Prithvi 13-class index -> coarse.
CDL_TO_COARSE = {
    0: _C["veg"], 1: _C["tree"], 2: _C["cropland"], 3: _C["cropland"], 4: _C["wetland"],
    5: _C["built"], 6: _C["water"], 7: _C["cropland"], 8: _C["cropland"], 9: _C["cropland"],
    10: _C["cropland"], 11: _C["cropland"], 12: _C["other"],
}
# ESA WorldCover value -> coarse.
WC_TO_COARSE = {
    10: _C["tree"], 20: _C["veg"], 30: _C["veg"], 40: _C["cropland"], 50: _C["built"],
    60: _C["built"], 70: _C["other"], 80: _C["water"], 90: _C["wetland"], 95: _C["wetland"], 100: _C["veg"],
}


def _remap(arr: np.ndarray, table: dict, default: int) -> np.ndarray:
    out = np.full(arr.shape, default, dtype=np.uint8)
    for k, v in table.items():
        out[arr == k] = v
    return out


def _resize_nn(arr: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((w_px, h_px), Image.NEAREST))


def landcover_agreement(pred_cls: np.ndarray, wc: np.ndarray, bounds) -> dict:
    w_px, h_px = eval_grid(bounds, target=256)
    pred_c = _remap(_resize_nn(pred_cls, w_px, h_px), CDL_TO_COARSE, _C["other"])
    wc_c = _remap(_resize_nn(wc, w_px, h_px), WC_TO_COARSE, _C["other"])

    overall = float((pred_c == wc_c).mean())
    per_class = []
    for name, ci in _C.items():
        pm, rm = pred_c == ci, wc_c == ci
        if not pm.any() and not rm.any():
            continue
        inter = int((pm & rm).sum())
        union = int((pm | rm).sum())
        per_class.append({
            "class": name,
            "iou": round(inter / union, 3) if union else 0.0,
            "pred_pct": round(100 * pm.mean(), 1),
            "ref_pct": round(100 * rm.mean(), 1),
        })
    per_class.sort(key=lambda d: d["ref_pct"], reverse=True)
    return {"overall_agreement": round(overall, 3), "per_class": per_class}
