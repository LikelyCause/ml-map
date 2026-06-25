"""Reference ('ground truth') data fetchers for evaluation.

- Buildings/roads: OpenStreetMap via the Overpass API (free, no key).
- Land cover: ESA WorldCover (10 m global, 11 classes) via the Planetary Computer.

These are imperfect references (OSM completeness varies; WorldCover is 10 m and a
different taxonomy than the crop model) — but they let us put real numbers on how
the foundation models perform, which is the point.
"""
from __future__ import annotations

import requests
from shapely.geometry import LineString, Polygon, mapping

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
_HEADERS = {"User-Agent": "swath/0.1 (geospatial model demo)", "Accept": "application/json"}


def _overpass(query: str) -> dict:
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=_HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = e
    raise RuntimeError(f"Overpass request failed: {last}")


def osm_features(bbox: list[float], kind: str) -> dict:
    """Fetch OSM buildings or roads as a GeoJSON FeatureCollection.

    kind: 'buildings' -> way[building] polygons; 'roads' -> way[highway] lines.
    """
    w, s, e, n = bbox
    sel = '["building"]' if kind == "buildings" else '["highway"]'
    query = f"[out:json][timeout:50];(way{sel}({s},{w},{n},{e}););out geom;"
    data = _overpass(query)

    feats = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geom]
        try:
            if kind == "buildings":
                if len(coords) < 4:
                    continue
                g = Polygon(coords)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
            else:
                g = LineString(coords)
        except Exception:  # noqa: BLE001 - skip malformed geometry
            continue
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": {}})

    return {"type": "FeatureCollection", "features": feats}


def worldcover_classes(bbox: list[float]):
    """Fetch ESA WorldCover class raster clipped to bbox. Returns (array, classes-present)."""
    import numpy as np
    import planetary_computer as pc
    import pystac_client
    import rioxarray  # noqa: F401
    from rasterio.warp import transform_bounds

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    items = list(catalog.search(collections=["esa-worldcover"], bbox=bbox).items())
    if not items:
        raise ValueError("No ESA WorldCover tile for this AOI.")
    item = max(items, key=lambda it: it.properties.get("start_datetime", ""))
    da = rioxarray.open_rasterio(item.assets["map"].href, masked=True)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", da.rio.crs, *bbox)
    clip = da.rio.clip_box(minx, miny, maxx, maxy).rio.reproject("EPSG:4326")
    arr = np.nan_to_num(clip.values[0]).astype("uint8")
    return arr
