"""STAC ingest from the Microsoft Planetary Computer.

Phase 1 supports NAIP (0.3-1 m RGB+NIR aerial, US). Given an AOI bbox in
EPSG:4326 we search the catalog, mosaic the most recent date's tiles over the
AOI, reproject to WGS84, and write an 8-bit RGB PNG that the frontend overlays
as a MapLibre ImageSource. Returns metadata matching the `Chip` shape.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rioxarray  # noqa: F401 - registers the .rio accessor
from PIL import Image
from rasterio.warp import transform_bounds
from rioxarray.merge import merge_arrays

from backend.geo.chips import save_chip
from backend.progress import set_stage

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Guardrails so a careless AOI can't try to pull a gigapixel mosaic. These are
# NOT hardware limits — they bound ingest tile count, output chip size, and the
# downstream tiled-inference wall-time. Override via env (e.g. on a big-memory
# Mac) without touching code. Bigger MAX_SPAN_DEG at a fixed MAX_PX = coarser
# pixels (worse small-object detection); raise MAX_PX too to keep detail, at the
# cost of proportionally longer SAM/DINO inference.
MAX_SPAN_DEG = float(os.environ.get("SWATH_NAIP_MAX_SPAN_DEG", "0.05"))  # ~5.5 km per side
MAX_PX = int(os.environ.get("SWATH_NAIP_MAX_PX", "8192"))  # long-side px of the chip the models see


def _bbox_id(source: str, bbox: list[float]) -> str:
    key = f"{source}:" + ",".join(f"{c:.6f}" for c in bbox)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _validate(bbox: list[float]) -> None:
    w, s, e, n = bbox

    if not (e > w and n > s):
        raise ValueError("bbox must be [west, south, east, north] with east>west, north>south")
    
    if (e - w) > MAX_SPAN_DEG or (n - s) > MAX_SPAN_DEG:
        raise ValueError(f"AOI too large; keep each side under ~{MAX_SPAN_DEG} deg for a demo chip")


def ingest_naip(bbox: list[float]) -> dict:
    """Fetch a NAIP chip for the AOI. bbox = [west, south, east, north] (EPSG:4326)."""

    _validate(bbox)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chip_id = _bbox_id("naip", bbox)
    png_path = DATA_DIR / f"{chip_id}_raw.png"

    set_stage("ingest", "Searching NAIP catalog…")
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = list(catalog.search(collections=["naip"], bbox=bbox, limit=50).items())
    
    if not items:
        raise ValueError("No NAIP imagery for this AOI (NAIP is US-only).")

    # Mosaic ALL tiles overlapping the AOI, newest first, so recent imagery wins in
    # overlaps and older tiles fill the gaps. A single capture date often doesn't
    # cover the whole box (NAIP is flown in date-bounded strips), which otherwise
    # leaves the chip smaller than the drawn AOI. merge_arrays keeps the first
    # array's valid pixels, so newest-first => newest on top.
    items.sort(key=lambda it: it.properties.get("datetime", ""), reverse=True)
    tiles = items
    dates = sorted({it.properties.get("datetime", "")[:10] for it in tiles})
    latest_date = dates[-1] if dates else ""

    target_crs = None
    pieces = []

    for i, it in enumerate(tiles, 1):
        set_stage("ingest", f"Downloading NAIP tile {i}/{len(tiles)}…")
        da = rioxarray.open_rasterio(it.assets["image"].href, masked=True)
    
        if target_crs is None:
            target_crs = da.rio.crs
        elif da.rio.crs != target_crs:
            da = da.rio.reproject(target_crs)
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", target_crs, *bbox)
    
        try:
            pieces.append(da.rio.clip_box(minx, miny, maxx, maxy))
        except Exception:  # noqa: BLE001 - tile doesn't actually overlap the AOI
            continue
    
    if not pieces:
        raise ValueError("AOI did not overlap any NAIP tile.")

    set_stage("ingest", f"Mosaicking {len(pieces)} tile(s) + reprojecting…")
    mosaic = (pieces[0] if len(pieces) == 1 else merge_arrays(pieces)).rio.reproject("EPSG:4326")

    marr = mosaic.values  # (bands, H, W); NAIP is 8-bit R,G,B,NIR
    rgb = np.nan_to_num(marr[:3]).astype(np.uint8)
    img = Image.fromarray(np.transpose(rgb, (1, 2, 0)), "RGB")
    
    if max(img.size) > MAX_PX:
        img.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)

    # De-stretch for the models: EPSG:4326 pixels are degree-linear, so at latitude
    # φ the image is stretched horizontally by 1/cos(φ) vs true ground (~1.31x at
    # 40°N). Compress width to cos(φ) so a square on the ground is square in pixels —
    # Grounding DINO and SAM were trained on undistorted imagery. from_bounds reads
    # the PNG's actual size, so georeferencing stays exact; the frontend pins the
    # overlay to the WGS84 bounds either way, so display is unaffected.
    lat_c = math.radians((bbox[1] + bbox[3]) / 2)
    cw = max(1, round(img.size[0] * math.cos(lat_c)))
    if cw != img.size[0]:
        img = img.resize((cw, img.size[1]), Image.LANCZOS)

    # NDVI sidecar from NAIP's NIR band (band 4) so the buildings filter can reject
    # vegetation (tree canopy / lawns) that SAM otherwise masks as "buildings".
    # uint8-encoded ((NDVI+1)*127.5), resized to the PNG's de-stretched layout and
    # capped for compact storage; the filter resizes it to each mask's resolution.
    if marr.shape[0] >= 4:
        red_f, nir_f = marr[0].astype(np.float32), marr[3].astype(np.float32)
        ndvi = np.nan_to_num((nir_f - red_f) / (nir_f + red_f + 1e-6))
        ndvi_u8 = ((ndvi + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        nw, nh = img.size
        if max(nw, nh) > 2048:
            sc = 2048 / max(nw, nh)
            nw, nh = max(1, round(nw * sc)), max(1, round(nh * sc))
        ndvi_small = Image.fromarray(ndvi_u8, "L").resize((nw, nh), Image.BILINEAR)
        np.save(DATA_DIR / f"{chip_id}_ndvi.npy", np.asarray(ndvi_small, dtype=np.uint8))

    set_stage("ingest", "Writing chip…")
    img.save(png_path)

    w, s, e, n = mosaic.rio.bounds()
    
    meta = {
        "id": chip_id,
        "bounds": [w, s, e, n],
        "raw_url": f"/data/{png_path.name}",
        "annotated_url": None,
        "source": "naip",
        "datetime": latest_date,
        "gsd": tiles[0].properties.get("gsd"),
        "size_px": list(img.size),
        "note": f"NAIP mosaic, {len(pieces)} tile(s) over {len(dates)} date(s)",
    }
    
    save_chip(meta)
    set_stage("done", "Imagery ready")
    return meta
