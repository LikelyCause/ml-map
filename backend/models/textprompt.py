"""Open-vocabulary segmentation: Grounding DINO + SAM.

Grounding DINO detects bounding boxes for an arbitrary text prompt; SAM then
masks each box. Zero-shot — type any object ("solar panel", "swimming pool",
"car") and segment it. Output is WGS84 GeoJSON with label + confidence.

This is also the stack the buildings task will reuse (prompt="building") to
replace the weaker segment-everything approach.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape

from backend.models.device import detector_device
from backend.progress import set_stage

from .buildings import (
    MAX_AREA_M2,
    MAX_ASPECT,
    MAX_IMAGE_FRACTION,
    MIN_AREA_M2,
    MIN_EXTENT,
    NDVI_VEG_THRESH,
    ndvi_for_masks,
)
from .sam import masks_from_boxes

# Detection runs at full native resolution (up to this cap) so small/medium
# overhead objects keep their detail. SAM masking is capped lower to bound
# CPU RAM during mask upscaling (SAM's image encoder works at 1024 internally
# regardless, so higher input mainly costs memory, not boundary quality).
DETECT_MAX_PX = 4096
SAM_MAX_PX = 2048
MAX_BOXES = 500
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Tiled detection (SAHI-style): overhead objects are small/oddly-scaled for a
# detector trained on natural images. Slicing into small overlapping tiles lets
# the detector's internal resize upscale objects, finding far more of them.
# Smaller tiles + more overlap = higher recall, more tiles (slower).
TILE_PX = 512
TILE_OVERLAP = 0.35
NMS_IOU = 0.5

# Roads are linear; box-prompted SAM otherwise masks the dominant region in each
# box (a field/lawn parcel). Require an elongated footprint to keep roads and drop
# compact blobs. Measured separation: real roads elongation ≳13, field blobs ≲3.
ROAD_MIN_ELONGATION = 4.0

_DETECTORS: dict[str, tuple] = {}


def _ground_shape(poly, bounds):
    """(area_m2, image_fraction, extent, elongation) for a lon/lat polygon, with
    cos(lat) correction so meters and elongation aren't distorted by latitude."""
    w, s, e, n = bounds
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians((s + n) / 2))
    area_m2 = poly.area * m_lat * m_lon
    chip_deg_area = (e - w) * (n - s)
    image_fraction = poly.area / chip_deg_area if chip_deg_area else 0.0
    extent = poly.area / poly.envelope.area if poly.envelope.area else 0.0
    xs, ys = poly.minimum_rotated_rectangle.exterior.coords.xy
    edges = sorted(
        math.hypot((xs[i + 1] - xs[i]) * m_lon, (ys[i + 1] - ys[i]) * m_lat)
        for i in range(len(xs) - 1)
    )
    edges = [d for d in edges if d > 0]
    elongation = edges[-1] / edges[0] if len(edges) >= 2 else 99.0
    return area_m2, image_fraction, extent, elongation


def _passes_task_filter(poly, bounds, task) -> bool:
    """Shape gate for the DINO+SAM output. Free-text (None/'textprompt') keeps
    everything (honest zero-shot); roads require linearity; buildings reuse the
    segment-everything footprint filters so DINO+SAM stops emitting block blobs."""
    if task not in ("roads", "buildings"):
        return True
    area_m2, frac, extent, elong = _ground_shape(poly, bounds)
    if task == "roads":
        return elong >= ROAD_MIN_ELONGATION
    return (
        frac <= MAX_IMAGE_FRACTION
        and MIN_AREA_M2 <= area_m2 <= MAX_AREA_M2
        and extent >= MIN_EXTENT
        and elong <= MAX_ASPECT
    )


def _starts(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    ss = list(range(0, size - tile + 1, step))
    if ss[-1] != size - tile:
        ss.append(size - tile)
    return ss


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(dets, iou_thresh: float):
    kept = []
    for box, score, label in sorted(dets, key=lambda d: d[1], reverse=True):
        if all(_iou(box, k[0]) < iou_thresh for k in kept):
            kept.append((box, score, label))
    return kept


def _get_detector(det_id: str):
    if det_id not in _DETECTORS:
        set_stage("model", f"Loading detector {det_id}…")
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = detector_device()
        proc = AutoProcessor.from_pretrained(det_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(device).eval()
        _DETECTORS[det_id] = (model, proc, device)
    return _DETECTORS[det_id]


def _normalize_prompt(prompt: str) -> str:
    # Grounding DINO expects lowercase, period-separated phrases ending in '.'.
    text = ". ".join(p.strip().lower() for p in prompt.split(",") if p.strip())
    return text if text.endswith(".") else text + "."


def _detect_image(det_id: str, image, text: str):
    """Single-pass Grounding DINO detection. Returns [(box_xyxy, score, label), ...]."""
    model, proc, device = _get_detector(det_id)
    inputs = proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    w_px, h_px = image.size
    results = proc.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h_px, w_px)],
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results.get("text_labels") or results.get("labels") or [""] * len(boxes)
    return list(zip(boxes, scores, (str(x) for x in labels)))


def _detect(det_id: str, image, prompt: str):
    """Tiled Grounding DINO detection over the whole image, merged with NMS."""
    text = _normalize_prompt(prompt)
    w_px, h_px = image.size
    if max(w_px, h_px) <= TILE_PX:
        dets = _detect_image(det_id, image, text)
    else:
        step = int(TILE_PX * (1 - TILE_OVERLAP))
        ys, xs = _starts(h_px, TILE_PX, step), _starts(w_px, TILE_PX, step)
        total = len(ys) * len(xs)
        dets = []
        k = 0
        for y in ys:
            for x in xs:
                k += 1
                set_stage("infer", f"Detecting '{prompt}' (tile {k}/{total})…")
                crop = image.crop((x, y, min(x + TILE_PX, w_px), min(y + TILE_PX, h_px)))
                for box, score, label in _detect_image(det_id, crop, text):
                    dets.append(([box[0] + x, box[1] + y, box[2] + x, box[3] + y], score, label))
        dets = _nms(dets, NMS_IOU)
    return sorted(dets, key=lambda d: d[1], reverse=True)[:MAX_BOXES]


def segment_by_text(png_path, bounds, det_id: str, seg_id: str, prompt: str, task: str | None = None) -> dict:
    base = Image.open(png_path).convert("RGB")

    # Detect at full native resolution (down only if above the cap).
    det_img = base
    if max(base.size) > DETECT_MAX_PX:
        det_img = base.copy()
        det_img.thumbnail((DETECT_MAX_PX, DETECT_MAX_PX), Image.LANCZOS)
    dW, dH = det_img.size

    dets = _detect(det_id, det_img, prompt)
    if not dets:
        return {"type": "FeatureCollection", "features": []}

    # Mask at a (possibly lower) capped resolution to bound memory; scale boxes.
    sam_img = base
    if max(base.size) > SAM_MAX_PX:
        sam_img = base.copy()
        sam_img.thumbnail((SAM_MAX_PX, SAM_MAX_PX), Image.LANCZOS)
    w_px, h_px = sam_img.size
    sx, sy = w_px / dW, h_px / dH
    boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b, _, _ in dets]

    masks = masks_from_boxes(seg_id, sam_img, boxes)
    set_stage("infer", f"Vectorizing {len(dets)} detection(s)…")
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w_px, h_px)
    simplify_tol = 0.5 / 111320.0  # ~0.5 m in degrees
    veg_ndvi = ndvi_for_masks(png_path, w_px, h_px) if task == "buildings" else None

    features = []
    for (_, score, label), m in zip(dets, masks):
        m = np.asarray(m, dtype=bool)
        if m.shape != (h_px, w_px) or m.sum() == 0:
            continue
        if veg_ndvi is not None and float(veg_ndvi[m].mean()) > NDVI_VEG_THRESH:
            continue
        geoms = [
            shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=transform) if v == 1
        ]
        if not geoms:
            continue
        poly = max(geoms, key=lambda g: g.area)
        if poly.is_empty or poly.area <= 0:
            continue
        if not _passes_task_filter(poly, bounds, task):
            continue
        poly = poly.simplify(simplify_tol, preserve_topology=True)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"label": label, "score": round(float(score), 3)},
            }
        )

    return {"type": "FeatureCollection", "features": features}
