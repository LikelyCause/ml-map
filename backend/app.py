"""Swath backend (FastAPI).

Serves AOI ingest (NAIP / Sentinel-2 via the Planetary Computer), the model zoo
(SAM, Grounding DINO, Prithvi-EO, Clay, fine-tuned heads) behind /infer, and
evaluation vs OSM / ESA WorldCover behind /evaluate — plus /progress and static
chip serving for the React/MapLibre split view.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.eval.evaluate import evaluate
from backend.geo.demo import ensure_demo
from backend.ingest.sentinel2 import ingest_sentinel2
from backend.ingest.stac import ingest_naip
from backend.models.infer import run_inference
from backend.models.registry import list_models, task_source
from backend.progress import get_progress

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Swath", version="0.1.0")

# Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/api/progress")
def progress() -> dict:
    """Live status of the current long-running operation (polled by the UI)."""
    return get_progress()


@app.get("/api/health")
def health() -> dict:
    gpu = False
    device = "cpu"
    try:
        from backend.models.device import get_device  # noqa: PLC0415 - optional until model phases

        device = get_device().type  # "cuda" | "mps" | "cpu"
        gpu = device != "cpu"
    except Exception:
        pass
    return {"status": "ok", "gpu": gpu, "device": device}


@app.get("/api/demo")
def demo() -> dict:
    """Return metadata for the Phase 0 synthetic demo chip."""
    return ensure_demo()


class IngestRequest(BaseModel):
    bbox: list[float]  # [west, south, east, north] in EPSG:4326
    source: str = "naip"


@app.post("/api/ingest")
def ingest(req: IngestRequest) -> dict:
    """Fetch real imagery for an AOI from the Planetary Computer."""
    try:
        if req.source == "naip":
            return ingest_naip(req.bbox)
        if req.source == "sentinel2":
            return ingest_sentinel2(req.bbox)
        raise HTTPException(400, f"Unsupported source '{req.source}'.")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001 - surface upstream/network errors to the UI
        raise HTTPException(502, f"Ingest failed: {e}")


@app.get("/api/models")
def models(task: str) -> dict:
    return {"task": task, "models": list_models(task), "source": task_source(task)}


class InferRequest(BaseModel):
    chip_id: str
    task: str
    model_id: str
    prompt: str | None = None


@app.post("/api/infer")
def infer(req: InferRequest) -> dict:
    try:
        return run_inference(req.chip_id, req.task, req.model_id, req.prompt)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001 - surface model/runtime errors to the UI
        raise HTTPException(500, f"Inference failed: {e}")


@app.post("/api/evaluate")
def evaluate_endpoint(req: InferRequest) -> dict:
    """Score a model's prediction against reference data (OSM / ESA WorldCover)."""
    try:
        return evaluate(req.chip_id, req.task, req.model_id, req.prompt)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001 - surface upstream/runtime errors to the UI
        raise HTTPException(502, f"Evaluation failed: {e}")
