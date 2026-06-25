"""SAM 'segment everything' mask generation via the HF mask-generation pipeline.

Pipelines are cached per checkpoint and kept warm on the accelerator. SAM
checkpoints are a few hundred MB (base) to ~2.5 GB (huge); the 16 GB 4080 holds
them comfortably, as does an Apple Silicon Mac with enough unified memory.
"""
from __future__ import annotations

import torch
from transformers import pipeline

from backend.models.device import get_device, pipeline_device
from backend.progress import set_stage

_PIPES: dict[str, object] = {}
_SAM_MODELS: dict[str, tuple] = {}


def get_mask_pipeline(hf_id: str):
    if hf_id not in _PIPES:
        set_stage("model", f"Loading {hf_id}…")
        _PIPES[hf_id] = pipeline("mask-generation", model=hf_id, device=pipeline_device())

    return _PIPES[hf_id]


def generate_masks(hf_id: str, image, points_per_side: int = 32):
    """Return a list of HxW boolean masks for everything SAM finds in `image`."""

    pipe = get_mask_pipeline(hf_id)
    set_stage("infer", "Segmenting everything with SAM…")

    with torch.inference_mode():
        out = pipe(
            image,
            points_per_side=points_per_side,
            points_per_batch=64,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
        )

    return out["masks"]  # list of numpy bool arrays at input resolution


def get_sam_model(seg_id: str):
    """Load (and cache) a SamModel + SamProcessor for box-prompted masking."""

    if seg_id not in _SAM_MODELS:
        set_stage("model", f"Loading {seg_id}…")
        from transformers import SamModel, SamProcessor

        device = get_device()
        model = SamModel.from_pretrained(seg_id).to(device).eval()
        proc = SamProcessor.from_pretrained(seg_id)
        _SAM_MODELS[seg_id] = (model, proc, device)
    
    return _SAM_MODELS[seg_id]


def masks_from_boxes(seg_id: str, image, boxes, batch: int = 64):
    """Mask each box with SAM. `boxes` = list of [x0,y0,x1,y1] pixel boxes.

    Boxes are processed in batches to bound memory at high resolution / large
    box counts (mask upscaling is the memory cost). Returns a list of HxW
    boolean masks aligned to `boxes`.
    """
    if not boxes:
        return []
    
    model, proc, device = get_sam_model(seg_id)
    out_masks = []

    for i in range(0, len(boxes), batch):
        chunk = boxes[i : i + batch]
        set_stage("infer", f"Masking regions {i + 1}-{i + len(chunk)} of {len(boxes)} with SAM…")
        inputs = proc(image, input_boxes=[chunk], return_tensors="pt")
        # SamProcessor emits float64 input_boxes; MPS has no float64, so
        # downcast any float64 tensors to float32 before the device move
        # (harmless on CUDA; pixel box coordinates do not need float64).
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.dtype == torch.float64:
                inputs[k] = v.to(torch.float32)
        inputs = inputs.to(device)

        with torch.inference_mode():
            out = model(**inputs, multimask_output=False)

        masks = proc.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        arr = masks[0].numpy().astype(bool)  # (nboxes, 1, H, W)
        out_masks.extend(arr[j, 0] for j in range(arr.shape[0]))
        
    return out_masks
