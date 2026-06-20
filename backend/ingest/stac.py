"""STAC ingest from the Microsoft Planetary Computer.

Phase 1 supports NAIP (0.3-1 m RGB+NIR aerial, US). Given an AOI bbox in
EPSG:4326 we search the catalog, mosaic the most recent date's tiles over the
AOI, reproject to WGS84, and write an 8-bit RGB PNG that the frontend overlays
as a MapLibre ImageSource. Returns metadata matching the `Chip` shape.
"""
from __future__ import annotations

import hashlib
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

# Guardrails so a careless AOI can't try to pull a gigapixel mosaic.
MAX_SPAN_DEG = 0.02  # ~2.2 km per side; plenty for a demo chip
MAX_PX = 4096  # downsample the long side of the output PNG to this (keep detail high)


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

    # Use the most recent capture date, and mosaic every tile from that date.
    latest_date = max(it.properties.get("datetime", "")[:10] for it in items)
    tiles = [it for it in items if it.properties.get("datetime", "").startswith(latest_date)]

    target_crs = None
    pieces = []
    for i, it in enumerate(tiles, 1):
        set_stage("ingest", f"Downloading NAIP tile {i}/{len(tiles)} ({latest_date})…")
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

    rgb = np.nan_to_num(mosaic.values[:3]).astype(np.uint8)  # NAIP is 8-bit, bands R,G,B,NIR
    img = Image.fromarray(np.transpose(rgb, (1, 2, 0)), "RGB")
    if max(img.size) > MAX_PX:
        img.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
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
        "note": f"NAIP mosaic, {len(pieces)} tile(s), {latest_date}",
    }
    save_chip(meta)
    set_stage("done", "Imagery ready")
    return meta
