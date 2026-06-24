"""Centralized accelerator selection.

Prefers CUDA (NVIDIA), then MPS (Apple Silicon Metal), then CPU. Keeping this in
one place means the same build runs on the dev RTX 4080 box and on Apple Silicon
without per-call-site `if cuda` branches.

On MPS, a handful of ops used by SAM / Grounding DINO are not yet implemented in
Metal. Set PYTORCH_ENABLE_MPS_FALLBACK=1 (see run.sh) so those transparently run
on CPU instead of raising NotImplementedError.
"""
from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def get_device() -> torch.device:
    """Best available accelerator as a torch.device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pipeline_device() -> torch.device:
    """Device argument for transformers.pipeline().

    The legacy integer convention (0=GPU, -1=CPU) only covers CUDA/CPU, so pass a
    torch.device instead — the pipeline accepts it and it carries 'mps' correctly.
    """
    return get_device()
