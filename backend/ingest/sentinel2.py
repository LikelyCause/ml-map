"""Sentinel-2 L2A multi-temporal ingest for the Prithvi land-cover model.

The crop/land model needs 3 timesteps x 6 bands (Blue, Green, Red, Narrow-NIR,
SWIR1, SWIR2) as 224x224 chips in HLS/S2 reflectance units. We pull 3 spread-out
low-cloud scenes from the Planetary Computer, stack the 6 bands, resample to
224x224, and also save a true-color RGB for display.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import torch
import torch.nn.functional as F
from odc.stac import load as odc_load
from PIL import Image

from backend.geo.chips import save_chip
from backend.progress import set_stage

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Prithvi band order -> Sentinel-2 L2A asset keys.
S2_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
MODEL_PX = 224
MAX_SPAN_DEG = 0.2  # land cover benefits from larger, varied AOIs (~20 km)


def _bbox_id(bbox: list[float]) -> str:
    key = "s2:" + ",".join(f"{c:.6f}" for c in bbox)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _validate(bbox: list[float]) -> None:
    w, s, e, n = bbox
    if not (e > w and n > s):
        raise ValueError("bbox must be [west, south, east, north]")
    if (e - w) > MAX_SPAN_DEG or (n - s) > MAX_SPAN_DEG:
        raise ValueError(f"AOI too large for land cover; keep each side under ~{MAX_SPAN_DEG} deg")


def _pick_three(items):
    """Pick 3 low-cloud scenes spread across time for multi-temporal input."""
    by_day = {}
    for it in items:
        day = it.properties.get("datetime", "")[:10]
        cc = it.properties.get("eo:cloud_cover", 100)
        if day not in by_day or cc < by_day[day].properties.get("eo:cloud_cover", 100):
            by_day[day] = it
    days = sorted(by_day)
    if len(days) < 3:
        return [by_day[d] for d in days] or None
    chosen = [days[0], days[len(days) // 2], days[-1]]  # earliest, middle, latest
    return [by_day[d] for d in chosen]


def ingest_sentinel2(bbox: list[float]) -> dict:
    _validate(bbox)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chip_id = _bbox_id(bbox)

    set_stage("ingest", "Searching Sentinel-2 catalog…")
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2023-01-01/2026-06-19",
            query={"eo:cloud_cover": {"lt": 15}},
        ).items()
    )
    if len(items) < 3:
        raise ValueError("Not enough clear Sentinel-2 scenes for this AOI.")

    chosen = _pick_three(items)
    dates = [it.properties["datetime"][:10] for it in chosen]
    set_stage("ingest", f"Loading 3 dates x 6 bands ({', '.join(dates)})…")

    res = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 512  # ~512 px display grid
    ds = odc_load(
        chosen, bands=S2_BANDS, bbox=bbox, crs="EPSG:4326",
        resolution=res, groupby="solar_day", chunks={},
    )
    # (time, band, y, x) in reflectance units
    stack = np.stack([ds[b].values for b in S2_BANDS], axis=1).astype(np.float32)
    stack = np.nan_to_num(stack)
    # Sentinel-2 processing baseline 04.00+ adds a +1000 BOA offset that HLS
    # (Prithvi's training data) does not have. Remove it to match the model's
    # expected reflectance units.
    stack = np.clip(stack - 1000.0, 0, None)
    if stack.shape[0] < 3:  # fewer distinct solar days than expected
        reps = int(np.ceil(3 / stack.shape[0]))
        stack = np.tile(stack, (reps, 1, 1, 1))[:3]
    stack = stack[:3]  # (3, 6, H, W)

    set_stage("ingest", "Building model input + preview…")
    # Model input: (6, 3, 224, 224) resampled.
    t = torch.from_numpy(stack)  # (3,6,H,W)
    t = F.interpolate(t, size=(MODEL_PX, MODEL_PX), mode="bilinear", align_corners=False)
    model_in = t.permute(1, 0, 2, 3).contiguous().numpy()  # (6,3,224,224)
    np.save(DATA_DIR / f"{chip_id}_lc.npy", model_in)

    # Display: true-color RGB from the latest scene (B04,B03,B02).
    latest = stack[-1]  # (6,H,W)
    rgb = np.stack([latest[2], latest[1], latest[0]], axis=-1)  # R,G,B
    rgb = np.clip(rgb / 3000.0 * 255.0, 0, 255).astype(np.uint8)
    png_path = DATA_DIR / f"{chip_id}_raw.png"
    Image.fromarray(rgb, "RGB").save(png_path)

    # bounds from the loaded grid (odc names coords longitude/latitude in EPSG:4326)
    xname = "longitude" if "longitude" in ds.coords else "x"
    yname = "latitude" if "latitude" in ds.coords else "y"
    xs, ys = ds[xname].values, ds[yname].values
    w, e = float(xs.min()), float(xs.max())
    s, n = float(ys.min()), float(ys.max())

    meta = {
        "id": chip_id,
        "bounds": [w, s, e, n],
        "raw_url": f"/data/{png_path.name}",
        "annotated_url": None,
        "source": "sentinel2",
        "dates": dates,
        "size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "note": f"Sentinel-2 L2A, 3 dates ({', '.join(dates)})",
    }
    save_chip(meta)
    set_stage("done", "Imagery ready")
    return meta
