"""Building-footprint extraction from SAM 'segment everything' masks.

SAM has no notion of 'building' — it segments every region. We approximate
footprints by keeping masks whose real-world area and shape look building-like,
then vectorize them to GeoJSON polygons in EPSG:4326. This is deliberately
zero-shot: the honest story is how far a general foundation model gets without
any building-specific training.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape

from backend.progress import set_stage

from .sam import generate_masks

# Cap inference resolution (SAM's image encoder works at 1024 px regardless).
MAX_INFER_PX = 1024

# Building-likeness filters.
MIN_AREA_M2 = 20.0
MAX_AREA_M2 = 8000.0
MAX_IMAGE_FRACTION = 0.25  # drop giant background/ground masks
MIN_EXTENT = 0.30  # poly area / bounding-rect area (buildings reasonably fill their box)
MAX_ASPECT = 6.0  # drop long thin masks (roads, shadows)


def _pixel_area_m2(bounds, w_px, h_px):
    w, s, e, n = bounds
    lat_c = math.radians((s + n) / 2)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_c)
    px_w_m = (e - w) * m_per_deg_lon / w_px
    px_h_m = (n - s) * m_per_deg_lat / h_px
    return abs(px_w_m * px_h_m)


def _aspect_ratio(poly) -> float:
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    edges = [
        math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(len(xs) - 1)
    ]
    edges = sorted(e for e in edges if e > 0)
    if len(edges) < 2 or edges[0] == 0:
        return 99.0
    return edges[-1] / edges[0]


def segment_buildings(png_path, bounds, hf_id: str) -> dict:
    img = Image.open(png_path).convert("RGB")
    if max(img.size) > MAX_INFER_PX:
        img.thumbnail((MAX_INFER_PX, MAX_INFER_PX), Image.LANCZOS)
    w_px, h_px = img.size
    masks = generate_masks(hf_id, img)

    set_stage("infer", f"Filtering + vectorizing {len(masks)} masks…")
    px_m2 = _pixel_area_m2(bounds, w_px, h_px)
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w_px, h_px)
    img_px = w_px * h_px

    features = []
    for m in masks:
        m = np.asarray(m, dtype=bool)
        if m.shape != (h_px, w_px):  # pipeline mask resolution sanity
            continue
        area_px = int(m.sum())
        if area_px == 0 or area_px / img_px > MAX_IMAGE_FRACTION:
            continue
        area_m2 = area_px * px_m2
        if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
            continue

        # Vectorize the mask; keep its largest connected polygon.
        geoms = [
            shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=transform) if v == 1
        ]
        if not geoms:
            continue
        poly = max(geoms, key=lambda g: g.area)
        if poly.is_empty or poly.area <= 0:
            continue

        extent = poly.area / poly.envelope.area if poly.envelope.area else 0
        if extent < MIN_EXTENT or _aspect_ratio(poly) > MAX_ASPECT:
            continue

        poly = poly.simplify(px_m2 ** 0.5 / 111320.0, preserve_topology=True)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"area_m2": round(area_m2, 1)},
            }
        )

    return {"type": "FeatureCollection", "features": features}
