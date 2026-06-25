"""Fine-tuned Prithvi-EO-2.0 segmentation tasks (terratorch checkpoints).

Two released, Apache-2.0 fine-tunes of Prithvi-EO-2.0-300M, loaded as native
terratorch SemanticSegmentationTask checkpoints:
  - burn-scar  (Prithvi-EO-2.0-300M-BurnScars,            UNetDecoder)
  - flood/water(Prithvi-EO-2.0-300M-TL-Sen1Floods11,      UperNetDecoder)

Both take the same 6-band Sentinel-2 stack the land-cover ingest already builds
(B02,B03,B04,B8A,B11,B12), a single 224x224 timestep, scaled to 0-1 reflectance
(x1e-4) then per-band normalized. Output is a 2-class mask we render as a single
colored overlay (positive class) + legend, like the land-cover task.

Normalization stats are taken from each model's shipped config / datamodule
(see docs/building-extraction-research.md). Weights download on first use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.geo.chips import load_chip
from backend.models.device import get_device
from backend.progress import set_stage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# model_id -> HF repo, checkpoint file, 0-1 reflectance means/stds, and the
# positive (rendered) class: name + RGBA color.
MODELS = {
    "prithvi2-burnscar": {
        "repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "ckpt": "Prithvi_EO_V2_300M_BurnScars.pt",
        "means": [0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
                  0.2323245113436119, 0.1972854853760658, 0.11944914225186566],
        "stds": [0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
                 0.07791732423672691, 0.08708738838140137, 0.07241979477437814],
        "classes": ["Not burned", "Burn scar"],
        "color": (255, 80, 30),
    },
    "prithvi2-flood": {
        "repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "ckpt": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "means": [0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015],
        "stds": [0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938],
        "classes": ["Not water", "Water / flood"],
        "color": (40, 120, 240),
    },
}

SCALE = 1e-4  # raw HLS/S2 reflectance -> 0-1, before normalization

_MODELS: dict[str, torch.nn.Module] = {}


def _get_model(model_id: str):
    if model_id not in _MODELS:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from terratorch.tasks import SemanticSegmentationTask  # noqa: PLC0415

        cfg = MODELS[model_id]
        set_stage("model", f"Loading {cfg['repo'].split('/')[-1]} (1.3 GB)…")
        ckpt = hf_hub_download(cfg["repo"], cfg["ckpt"])
        task = SemanticSegmentationTask.load_from_checkpoint(
            ckpt, map_location="cpu", strict=False
        )
        model = task.model.to(get_device()).eval()
        _MODELS[model_id] = model
    return _MODELS[model_id]


def _logits(out) -> torch.Tensor:
    """terratorch model output may be a tensor or a ModelOutput with .output."""
    return out.output if hasattr(out, "output") else out


def segment_finetuned(chip_id: str, model_id: str) -> dict:
    meta = load_chip(chip_id)
    cfg = MODELS[model_id]
    npy = DATA_DIR / f"{chip_id}_lc.npy"
    if not npy.exists():
        raise FileNotFoundError("Model input missing; re-fetch Sentinel-2 imagery.")

    arr = np.load(npy).astype(np.float32)[:, -1]  # latest timestep -> (6, 224, 224)
    means = np.array(cfg["means"], np.float32)[:, None, None]
    stds = np.array(cfg["stds"], np.float32)[:, None, None]
    x = (arr * SCALE - means) / stds
    t = torch.from_numpy(x).unsqueeze(0).to(get_device())  # (1, 6, 224, 224)

    model = _get_model(model_id)
    set_stage("infer", f"Running {model_id}…")
    with torch.inference_mode():
        pred = _logits(model(t)).argmax(1)[0].cpu().numpy().astype(np.uint8)  # (224, 224)

    set_stage("infer", "Colorizing…")
    h0, w0 = pred.shape
    rgba = np.zeros((h0, w0, 4), np.uint8)
    pos = pred == 1
    rgba[pos, 0], rgba[pos, 1], rgba[pos, 2] = cfg["color"]
    rgba[pos, 3] = 185
    disp_w, disp_h = meta.get("size_px", [w0, h0])
    overlay = Image.fromarray(rgba, "RGBA").resize((disp_w, disp_h), Image.NEAREST)
    out_png = DATA_DIR / f"{chip_id}_{model_id}.png"
    overlay.save(out_png)

    pos_pct = round(100.0 * float(pos.mean()), 1)
    legend = [
        {"class": cfg["classes"][1], "color": "#%02x%02x%02x" % cfg["color"], "pct": pos_pct},
        {"class": cfg["classes"][0], "color": "#444b5a", "pct": round(100 - pos_pct, 1)},
    ]
    return {
        "task": model_id,
        "overlay_url": f"/data/{out_png.name}",
        "bounds": meta["bounds"],
        "legend": legend,
    }
