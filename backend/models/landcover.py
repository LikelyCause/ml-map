"""Land-cover classification with the reconstructed Prithvi crop/land model.

Loads the saved 6x3x224x224 Sentinel-2 stack, normalizes per band, runs the
13-class segmentation, and renders a colorized georeferenced overlay + legend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from backend.geo.chips import load_chip
from backend.progress import set_stage

from .prithvi_model import BAND_MEANS, BAND_STDS, CLASSES, PrithviSeg, load_state_into

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification"
CKPT = "multi_temporal_crop_classification_Prithvi_100M.pth"

# 13-class colormap (RGB), index-aligned with CLASSES.
COLORS = [
    (150, 200, 120), (34, 120, 34), (240, 200, 40), (160, 170, 50), (80, 180, 170),
    (180, 90, 90), (40, 90, 200), (200, 170, 110), (230, 140, 40), (150, 110, 70),
    (230, 160, 200), (180, 60, 160), (200, 200, 200),
]

_MODEL: PrithviSeg | None = None


def _get_model() -> PrithviSeg:
    global _MODEL
    if _MODEL is None:
        set_stage("model", "Loading Prithvi land-cover model (1.7 GB)…")
        path = hf_hub_download(REPO, CKPT)
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        m = PrithviSeg().eval()
        load_state_into(m, sd)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = m.to(dev)
        _MODEL._dev = dev  # type: ignore[attr-defined]
    return _MODEL


def classify_landcover(chip_id: str) -> dict:
    meta = load_chip(chip_id)
    npy = DATA_DIR / f"{chip_id}_lc.npy"
    if not npy.exists():
        raise FileNotFoundError("Land-cover input missing; re-fetch Sentinel-2 imagery.")

    model = _get_model()
    dev = model._dev  # type: ignore[attr-defined]

    arr = np.load(npy).astype(np.float32)  # (6, 3, 224, 224)
    means = np.array(BAND_MEANS, dtype=np.float32)[:, None, None, None]
    stds = np.array(BAND_STDS, dtype=np.float32)[:, None, None, None]
    arr = (arr - means) / stds
    x = torch.from_numpy(arr).unsqueeze(0).to(dev)  # (1, 6, 3, 224, 224)

    set_stage("infer", "Classifying land cover (Prithvi)…")
    with torch.inference_mode():
        logits = model(x)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (224, 224)
    np.save(DATA_DIR / f"{chip_id}_landcover_cls.npy", pred)  # for evaluation

    set_stage("infer", "Colorizing + building legend…")
    h0, w0 = pred.shape
    rgba = np.zeros((h0, w0, 4), dtype=np.uint8)
    counts = np.bincount(pred.ravel(), minlength=len(CLASSES))
    for c in range(len(CLASSES)):
        m = pred == c
        if m.any():
            rgba[m, 0], rgba[m, 1], rgba[m, 2] = COLORS[c]
            rgba[m, 3] = 175

    # Upscale to display size for a crisp overlay.
    disp_w, disp_h = meta.get("size_px", [w0, h0])
    overlay = Image.fromarray(rgba, "RGBA").resize((disp_w, disp_h), Image.NEAREST)
    out_png = DATA_DIR / f"{chip_id}_landcover.png"
    overlay.save(out_png)

    total = int(counts.sum())
    legend = [
        {"class": CLASSES[c], "color": "#%02x%02x%02x" % COLORS[c],
         "pct": round(100.0 * counts[c] / total, 1)}
        for c in np.argsort(counts)[::-1]
        if counts[c] > 0
    ]
    return {
        "task": "landcover",
        "overlay_url": f"/data/{out_png.name}",
        "bounds": meta["bounds"],
        "legend": legend,
    }
