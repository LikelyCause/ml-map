"""Inference dispatch + result caching.

Maps a (chip, task, model) request to the right model wrapper and returns
GeoJSON. Results are cached on disk keyed by chip/task/model so re-running the
same combo is instant.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from backend.geo.chips import chip_png_path, load_chip
from backend.progress import set_stage

from .buildings import segment_buildings
from .landcover import classify_landcover
from .registry import resolve_model
from .textprompt import segment_by_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Bump when the inference pipeline changes so stale cached results are ignored.
PIPE_VERSION = "v4-ndvi"

# Fixed prompts for tasks that use the DINO+SAM stack but don't take user text.
TASK_PROMPTS = {
    "buildings": "building.",
    "roads": "road. street. highway.",
}


def run_inference(chip_id: str, task: str, model_id: str, prompt: str | None = None) -> dict:
    set_stage("infer", "Preparing…")
    meta = load_chip(chip_id)
    model = resolve_model(task, model_id)

    # Land cover is a raster task (colorized class overlay + legend, not vectors).
    if task == "landcover":
        cache = DATA_DIR / f"{chip_id}_landcover_{model_id}_{PIPE_VERSION}.json"
        if cache.exists():
            return {**json.loads(cache.read_text()), "cached": True}
        t0 = time.time()
        res = classify_landcover(chip_id)
        res.update(model_id=model_id, cached=False, ms=int((time.time() - t0) * 1000))
        cache.write_text(json.dumps(res))
        set_stage("done", "Land cover ready")
        return res

    # Resolve the effective text prompt: user text for textprompt, else the
    # task's fixed prompt (roads/buildings-via-DINO). SAM-everything ignores it.
    if "detector" in model:
        if task == "textprompt":
            if not prompt or not prompt.strip():
                raise ValueError("Text-prompt task requires a 'prompt'.")
            eff_prompt = prompt.strip()
        else:
            eff_prompt = TASK_PROMPTS.get(task)
            if not eff_prompt:
                raise ValueError(f"No fixed prompt configured for task '{task}'.")
    else:
        eff_prompt = None

    # Different prompts must not share a cache entry.
    suffix = ""
    if eff_prompt:
        suffix = "_" + hashlib.sha1(eff_prompt.lower().encode()).hexdigest()[:8]
    cache = DATA_DIR / f"{chip_id}_{task}_{model_id}{suffix}_{PIPE_VERSION}.geojson"

    if cache.exists():
        gj = json.loads(cache.read_text())
        return {"task": task, "model_id": model_id, "cached": True,
                "count": len(gj["features"]), "geojson": gj}

    png = chip_png_path(chip_id)
    if not png.exists():
        raise FileNotFoundError("Chip imagery missing; re-fetch the AOI.")

    t0 = time.time()
    if "detector" in model:
        gj = segment_by_text(png, meta["bounds"], model["detector"], model["segmenter"], eff_prompt, task=task)
    elif "hf" in model:
        gj = segment_buildings(png, meta["bounds"], model["hf"])
    else:
        raise ValueError(f"Model '{model_id}' has no detector or hf checkpoint.")
    ms = int((time.time() - t0) * 1000)

    cache.write_text(json.dumps(gj))
    set_stage("done", f"{len(gj['features'])} features")
    return {"task": task, "model_id": model_id, "cached": False, "ms": ms,
            "count": len(gj["features"]), "geojson": gj}
