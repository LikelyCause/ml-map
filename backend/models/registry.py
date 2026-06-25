"""Catalog of HuggingFace models available per task.

Drives the frontend Model dropdown and maps a (task, model_id) to a concrete
HF checkpoint. Add entries here as tasks come online in later phases.
"""
from __future__ import annotations

# Reused DINO+SAM combos (open-vocab detect → mask). Used by textprompt, roads,
# and the buildings task (with a fixed prompt per task — see TASK_PROMPTS in infer).
_DINO_SAM = [
    {
        "id": "gdino-tiny-sam-base",
        "label": "G-DINO tiny + SAM base (fast)",
        "detector": "IDEA-Research/grounding-dino-tiny",
        "segmenter": "facebook/sam-vit-base",
    },
    {
        "id": "gdino-base-sam-large",
        "label": "G-DINO base + SAM large (best)",
        "detector": "IDEA-Research/grounding-dino-base",
        "segmenter": "facebook/sam-vit-large",
    },
]

TASKS: dict[str, dict] = {
    "buildings": {
        "label": "Building footprints",
        "models": [
            {"id": "sam-vit-base", "hf": "facebook/sam-vit-base", "label": "SAM everything · ViT-Base (fast)"},
            {"id": "sam-vit-large", "hf": "facebook/sam-vit-large", "label": "SAM everything · ViT-Large"},
            {"id": "sam-vit-huge", "hf": "facebook/sam-vit-huge", "label": "SAM everything · ViT-Huge"},
            *_DINO_SAM,  # detect "building" → SAM: cleaner per-building footprints
        ],
    },
    "roads": {
        "label": "Roads / lines of communication",
        "models": list(_DINO_SAM),
    },
    "textprompt": {
        "label": "Text-prompt segmentation",
        "models": list(_DINO_SAM),
    },
    "landcover": {
        "label": "Land cover",
        "source": "sentinel2",  # uses Sentinel-2, not NAIP
        "models": [
            {"id": "prithvi-100m-crop", "label": "Prithvi-EO-1.0 100M (13-class)", "landcover": True},
        ],
    },
    "embeddings": {
        "label": "Foundation embeddings (unsupervised)",
        "source": "sentinel2",  # GeoFM encoders consume the same 6-band S2 stack
        "models": [
            {"id": "prithvi-eo-2-300", "label": "Prithvi-EO-2.0 300M (IBM/NASA)", "embeddings": True},
            {"id": "clay-v1-base", "label": "Clay v1 base (Clay Foundation)", "embeddings": True},
        ],
    },
    "burnscar": {
        "label": "Burn scars (wildfire)",
        "source": "sentinel2",
        "models": [
            {"id": "prithvi2-burnscar", "label": "Prithvi-EO-2.0 300M · BurnScars", "finetuned": True},
        ],
    },
    "flood": {
        "label": "Flood / surface water",
        "source": "sentinel2",
        "models": [
            {"id": "prithvi2-flood", "label": "Prithvi-EO-2.0 300M · Sen1Floods11", "finetuned": True},
        ],
    },
}


def list_models(task: str) -> list[dict]:
    return TASKS.get(task, {}).get("models", [])


def task_source(task: str) -> str:
    """Imagery source for a task ('naip' for high-res, 'sentinel2' for land cover)."""
    return TASKS.get(task, {}).get("source", "naip")


def resolve_model(task: str, model_id: str) -> dict:
    for m in list_models(task):
        if m["id"] == model_id:
            return m
    raise KeyError(f"No model '{model_id}' for task '{task}'")
