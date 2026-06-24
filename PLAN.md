# ML‑Map — Project Plan & Handoff

A portfolio app demonstrating **geospatial foundation models** for inference (no
training) on remotely sensed imagery. Split view: raw imagery left, model
annotations right. Pick an area → fetch high‑res imagery → run a model → compare.

Last updated: 2026‑06‑19. **Status: ALL PHASES (0–5) COMPLETE & verified.**

### Phase 5 summary (evaluation + polish)
- **Reference data:** `backend/eval/reference.py` — OSM buildings/roads via Overpass
  (UA header + mirror fallback), ESA WorldCover via Planetary Computer STAC.
- **Metrics:** `backend/eval/metrics.py` — rasterize pred+ref to a common grid →
  IoU/precision/recall/F1 (roads buffer ref lines ~6 m); land cover maps both the
  13 CDL classes and WorldCover to a coarse scheme → agreement + per-class IoU.
- **`/api/evaluate`** (`backend/eval/evaluate.py`) reuses cached predictions,
  fetches reference, scores. Frontend: **Evaluate** button, metrics panel,
  cyan reference overlay on the left map (vs orange prediction on the right).
- **Verified numbers:** Buildings DINO+SAM IoU 0.53 / F1 0.69 vs SAM-everything
  IoU 0.41 / F1 0.58 (158 OSM footprints); Roads precision 0.66 / recall 0.21
  (506 OSM roads); Land cover agreement varies by AOI (crop model over-predicts
  cropland off-farmland).
- **README** rewritten portfolio-grade with results table + `docs/*_example.png`.

### Phase 4 summary (roads + land cover + progress UI + hi-res)
- **Buildings-fix flipped on:** buildings task now offers SAM-everything (3) + DINO+SAM (2). DINO+SAM ("building") gives clean per-building footprints.
- **Roads:** DINO+SAM with fixed prompt "road. street. highway.". Honest hard case — finds prominent arterials/rail (rail = valid "line of communication") but misses the residential grid; some false positives. Documented limitation.
- **Hi-res tiling:** detection now runs at full native resolution (`DETECT_MAX_PX=4096`, 512px tiles, 35% overlap) — ~3x more detections. SAM masking batched (64) + capped at 2048px to bound memory. Ingest `MAX_PX` raised to 4096.
- **Live progress UI:** `backend/progress.py` (in-process tracker) + `GET /api/progress`, pipeline instrumented with `set_stage()`, frontend polls every 400ms and shows a status banner.
- **Land cover (the big one):**
  - Model is the **mmseg-era** Prithvi-EO-1.0-100M crop classification (.pth). Rebuilt the exact architecture in plain PyTorch (`backend/models/prithvi_model.py`) — TemporalViTEncoder (depth 6!) + ConvTransformerTokensToEmbeddingNeck + FCNHead — loads the legacy weights with ZERO missing/unexpected keys. No mmcv/mmseg needed.
  - `backend/ingest/sentinel2.py`: 3 spread-out low-cloud S2 L2A dates × 6 bands (B02,B03,B04,B8A,B11,B12) via odc-stac → (6,3,224,224). **Critical:** subtract the +1000 S2 baseline-04.00 BOA offset to match HLS units (without it → garbage classes).
  - `backend/models/landcover.py`: normalize → classify → 13-class colorized RGBA overlay + legend (class %).
  - Verified over Iowa cropland: Corn 35% / Soy 20% / Winter Wheat 28% — sensible, field-aligned. **Caveat:** CDL-trained, US-cropland only; over non-farmland (foothills/cities) it mislabels — pick agricultural AOIs for the demo.
  - Frontend: land cover is a raster task (overlay + legend), uses Sentinel-2 source; task→source comes from `/api/models`.

---

## 1. Goal

- Split‑screen UI: remotely sensed imagery (left) vs. the same imagery annotated
  by ML (right), with synced pan/zoom.
- Four label tasks, each swappable between HuggingFace models via a dropdown:
  1. **Roads / lines of communication**
  2. **Building footprints**
  3. **General land‑cover classification**
  4. **Text‑prompt (open‑vocabulary) segmentation** — type any object, segment it
     *(this was the "designer's choice" 4th task)*
- A workflow to select a geographic area and one‑click fetch + ingest imagery.
- Inference only — learn how foundation models perform, no fine‑tuning.
- End product is portfolio‑grade.

---

## 2. Decisions locked (with rationale)

| Decision | Choice | Why |
|---|---|---|
| **Imagery source** | **NAIP + Sentinel‑2** via Microsoft Planetary Computer (STAC API) | NAIP = 0.3–1 m aerial (US) → buildings/roads visible. Sentinel‑2 = 10 m multispectral → proper input for land‑cover foundation models. One STAC API for both, free, no heavy auth. Trade‑off: NAIP is **US‑only**. |
| **UI / hosting** | **FastAPI backend + React/MapLibre frontend**, run **locally on the 4080** | Most professional/customizable for a portfolio. Local GPU runs every model with no cloud cost. |
| **4th task** | **Text‑prompt segmentation** (Grounding DINO + SAM) | Highest interactivity/wow‑factor; reuses the SAM stack from buildings. |
| **Land‑cover approach** | **Prithvi‑EO via terratorch** (fine‑tuned head) | Most authentic "geospatial foundation model" story. Pure foundation models output embeddings, not labels — terratorch supplies a fine‑tuned land‑cover head. |
| **Evaluation** | **Yes** — IoU/F1 vs. free reference data (OSM roads, open building footprints, ESA WorldCover) | Turns "pretty pictures" into "I evaluated foundation models against ground truth" — the strongest portfolio differentiator. |

### The honest framing (the portfolio narrative)
There is **no single foundation model** that does roads + buildings + land cover
zero‑shot well. The interesting story is the *comparison*: different model
families excel at different tasks, and the app demonstrates that. Roads are the
hardest zero‑shot task — that's an honest, valuable finding, not a failure.

---

## 3. Models per task (all HuggingFace, inference‑only)

| Task | Model(s) | Notes |
|---|---|---|
| Buildings | SAM2 (+ `segment-geospatial`/samgeo wrapper) | Works well on 0.3 m NAIP; visually strong |
| Roads / LOC | Grounding DINO + SAM ("road"/"highway" prompt) | Hardest zero‑shot — the honest comparison |
| Land cover | Prithvi‑EO‑2.0 via terratorch (and/or Clay v1.5) | Needs Sentinel‑2 multispectral input |
| Text‑prompt | Grounding DINO + SAM (open vocab) | Reuses SAM; type any object |

> Verify exact current HF checkpoint IDs at implementation time rather than
> hard‑coding from memory.

---

## 4. Architecture

```
React + MapLibre GL (frontend, Vite, port 5173)
 ├─ AOI panel: drag-draw bbox on left map
 ├─ split view: 2 synced MapLibre maps + AOI rectangle on both
 ├─ controls: task ▸ model ▸ (text-prompt) ▸ select area ▸ fetch
 └─ overlays: ImageSource (raster chips/masks) + GeoJSON (vectors)
        │  Vite proxies /api and /data ──► backend
        ▼
FastAPI (backend, port 8077, machine-learning venv, GPU)
 ├─ /api/health   → {status, gpu}
 ├─ /api/demo     → synthetic Phase 0 chip (kept for fallback/testing)
 ├─ /api/ingest   → STAC search + mosaic + clip + reproject → web PNG   [DONE: NAIP]
 ├─ /api/models   → registry of models per task                         [Phase 5]
 └─ /api/infer    → run model → GeoJSON and/or georeferenced mask PNG    [Phase 2+]
        ▼
Local model zoo (lazy-loaded, kept warm within 16 GB VRAM)              [Phase 2+]
 SAM2 · Grounding DINO+SAM · Prithvi/terratorch · Clay
```

**Design choice:** chips are small (single AOI), so we skip a tile server and
georeference result rasters directly as MapLibre `ImageSource` overlays
(PNG + WGS84 corner coords). Vectors go over as GeoJSON. Removes an
infrastructure layer while keeping alignment.

---

## 5. Environment (this machine)

- **OS:** Linux (WSL2). Working dir: `/home/primus/machine_learning`
- **Python:** pyenv virtualenv **`machine-learning`** = Python 3.12.13
  - **Use the venv python by absolute path** (pyenv shims don't re‑resolve in
    fresh non‑interactive shells):
    `/home/primus/.pyenv/versions/machine-learning/bin/python`
- **Node:** installed via **nvm** → Node v24.17.0 / npm 11.13.0
  - Source it first in any shell: `export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"`
- **GPU:** RTX 4080 Super, 16 GB VRAM, driver 610.62 (CUDA 12.x) — torch CUDA
  wheels work; system CUDA toolkit not required.
- **Apple Silicon:** runs natively on the PyTorch MPS backend. Install default
  PyPI torch wheels (no CUDA index). Accelerator is auto-selected at runtime
  (cuda → mps → cpu) in `backend/models/device.py`. `run.sh` exports
  `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops not yet implemented in Metal fall back to
  CPU. Models share unified memory, so prefer a Mac with ≥16 GB RAM.
- **Ports:** backend 8077, frontend 5173.

### Packages already in the venv (do NOT reinstall blindly — check first)
fastapi 0.137, uvicorn 0.49, numpy 2.4.6, pillow 12, rasterio 1.5, shapely 2.1,
geopandas 1.1, **pystac-client 0.9, planetary-computer 1.0, rioxarray 0.22,
odc-stac** (installed). torch / transformers / terratorch / SAM2 / GroundingDINO
**not yet installed** (Phase 2+).

> NOTE from the user: **always double‑check what's already installed before
> installing.** Use `pip list | grep -i <pkg>`.

---

## 6. How to run

```bash
cd /home/primus/machine_learning
./run.sh
# backend  → http://127.0.0.1:8077  (FastAPI; docs at /docs)
# frontend → http://127.0.0.1:5173  (open this)
```

`run.sh` sources nvm, starts uvicorn (with `--reload`) and `npm run dev`, and
kills both on exit. Frontend proxies `/api` + `/data`, so just open 5173.

Manual start (if needed):
```bash
VPY=/home/primus/.pyenv/versions/machine-learning/bin/python
$VPY -m uvicorn backend.app:app --host 127.0.0.1 --port 8077 --reload
# in another shell:
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"
cd frontend && npm run dev -- --host 127.0.0.1 --port 5173
```

---

## 7. File map

```
machine_learning/
├─ run.sh                     # start both dev servers
├─ README.md                  # portfolio-facing
├─ PLAN.md                    # this file
├─ .gitignore
├─ backend/
│  ├─ app.py                  # FastAPI: /api/health, /api/demo, /api/ingest
│  ├─ requirements.txt        # phased dependency list
│  ├─ geo/demo.py             # Phase 0 synthetic chip generator
│  ├─ ingest/stac.py          # NAIP STAC mosaic ingest (Planetary Computer)
│  ├─ models/                 # (empty) Phase 2+ model wrappers
│  └─ data/                   # cached chips + results (gitignored)
└─ frontend/
   ├─ vite.config.ts          # proxy /api + /data → :8077
   └─ src/
      ├─ App.tsx              # controls + draw/fetch workflow
      ├─ SplitMap.tsx         # two synced maps, AOI draw, chip overlays
      ├─ api.ts               # getHealth, ingestChip, getDemoChip, types
      ├─ App.css, index.css
```

---

## 8. What's built & verified

### Phase 0 — skeleton ✅
- FastAPI app, CORS, static `/data` serving, `/api/health`, `/api/demo`.
- Synthetic demo chip (Denver AOI) generator.
- React/MapLibre split view: two OSM basemaps, synced pan/zoom, chip overlaid
  as ImageSource (left raw, right raw+annotation placeholder).
- **Verified:** all endpoints HTTP 200; Vite proxy works; typecheck clean;
  user confirmed visual render (screenshot matched).

### Phase 1 — AOI ingest (NAIP) ✅
- `backend/ingest/stac.py`: searches NAIP on Planetary Computer, mosaics all
  tiles from the most recent capture date, clips to AOI in native UTM,
  reprojects to WGS84, downsamples long side to ≤2048 px, writes RGB PNG.
  Caches by bbox hash. AOI capped at 0.02°/side (HTTP 422 if exceeded).
- `POST /api/ingest {bbox:[w,s,e,n], source:"naip"}` → Chip metadata.
- Frontend: "Select area" → drag‑draw rectangle on left map (mirrored on right),
  "Fetch imagery (NAIP)" → ingest → real 0.3 m imagery in both panes; info bar
  shows scene/resolution/size.
- **Verified:** NAIP fetch ~7 s, 4‑tile mosaic, correct aspect (2048×1567),
  HTTP 200 via direct + proxy, 422 guard works, typecheck clean.
- **NOT yet eyeballed by the user:** the drag‑draw interaction and overlay‑vs‑
  basemap alignment in the browser. If the overlay looks shifted, suspect
  ImageSource corner coordinate order in `SplitMap.tsx:boundsToImageCoords`.

---

## 9. Remaining phases

### Phase 2 — Building footprints (SAM) ✅ DONE
- Installed PyTorch 2.6+cu124, transformers 5.12. `/api/health` → `gpu: true`.
- `models/registry.py` (task→HF models), `models/sam.py` (cached mask-generation
  pipeline), `models/buildings.py` (segment-everything → footprint filter →
  WGS84 GeoJSON), `models/infer.py` (dispatch + cache). `/api/models`, `/api/infer`.
- Frontend: live Model dropdown, Run model button, orange footprint overlay.

#### ⚠ Buildings quality finding (2026-06-18) + PLANNED fix (not yet built)
- **Both SAM ViT-Base and ViT-Huge performed poorly** on downtown NAIP. Visual
  diagnosis (rendered overlay): georeferencing is CORRECT, but SAM's automatic
  point grid segments at the **city-block level, not the building level** —
  whole blocks (roofs + courtyards + parking) become one polygon, plus spurious
  round/octagon blobs. Model size didn't fix it; the approach is the bottleneck
  (SAM has no concept of "building"). Area filter (≤8000 m²) lets whole ~6000 m²
  blocks through.
- **PLANNED FIX (decided, deferred):** swap to **Grounding DINO (text: "building"
  / "rooftop") → SAM box-prompted masks** = one clean mask per detected building.
  This is the **same stack as Phase 3** — once Phase 3 (text-prompt) is built,
  add a "Grounding DINO + SAM" entry to the *buildings* task using prompt
  "building", so the dropdown compares SAM-everything vs DINO+SAM directly.
  Optionally tune SAM (denser points + tighter filters) as a secondary compare.

### Phase 3 — Text‑prompt segmentation (Grounding DINO + SAM) ✅ DONE

Open-vocab: Grounding DINO detects boxes for any text prompt → SAM masks each →
WGS84 GeoJSON (props label + score). Frontend shows a Prompt input when task =
textprompt (Enter to run). Models: `gdino-tiny-sam-base` (fast),
`gdino-base-sam-large` (best). Prompt is part of the result cache key.

**KEY FINDING — tiled inference was essential.** Grounding DINO is trained on
ground-level photos and transfers poorly to overhead imagery: at full-scene
1024 px it returned only ~8 giant multi-block boxes for "building". Adding
**SAHI-style tiled detection** (512 px tiles, 30% overlap, NMS merge) in
`textprompt.py:_detect` raised that to ~46 well-aligned individual buildings in
~4 s. Small objects (cars at ~0.7 m/px) are still hard. This overhead-transfer
limitation is itself a good portfolio talking point.

**Files:** `sam.py` (`get_sam_model`, `masks_from_boxes`), `textprompt.py`
(tiled `_detect`, `_nms`, `segment_by_text`), `registry.py` (`resolve_model`
returns full dict; `textprompt` task), `infer.py` (dispatch + prompt cache key).

**API used (transformers 5.12), confirmed working:** see the verified-API notes
that were here previously — `post_process_grounded_object_detection(..., 
threshold=, text_threshold=, target_sizes=[(H,W)])`; SAM box masking via
`SamModel`/`SamProcessor` + `post_process_masks`.

**API verified this session (transformers 5.12 — use these exact calls):**
- Detector: `AutoModelForZeroShotObjectDetection` + `AutoProcessor` from
  `IDEA-Research/grounding-dino-tiny` (fast) or `-base` (better).
- Text query must be lowercase, period-separated, end with `.` (e.g. `"building."`).
- `processor.post_process_grounded_object_detection(outputs, input_ids,
  threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)])`
  → `results[0]["boxes"]` (xyxy px), `["scores"]`, `["text_labels"]`.
  ⚠ param is `threshold`, NOT `box_threshold`, in this version.
- SAM box masking: `SamModel` + `SamProcessor`;
  `proc(image, input_boxes=[[[x0,y0,x1,y1], ...]], return_tensors="pt")` →
  `model(**inputs, multimask_output=False)` →
  `proc.image_processor.post_process_masks(out.pred_masks.cpu(),
  inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())`
  → `masks[0]` shape `(nboxes, 1, H, W)` bool.
- Downscale chip to ≤1024 px for inference (as `buildings.py` does); georeference
  with the resized dims. Cap boxes to top ~150 by score.

**VRAM note:** the running backend caches every SAM size it has loaded; after
base+large+huge it sat at ~8.7 GB / 16 GB. Adding DINO+SAM is fine, but to be
safe long-term, add simple unload/limit logic (don't keep all SAM sizes
resident) before loading many models.

**Then immediately wire the deferred buildings fix:** add a buildings-task model
entry that calls the same DINO+SAM path with prompt `"building"`, so the
buildings dropdown compares SAM-everything vs DINO+SAM (see Phase 2 finding above).

### Phase 4 — Roads + Land cover
- **Roads:** Grounding DINO + SAM with "road/street/highway" prompt → GeoJSON
  (expect blobby results — document as the honest hard case). Optionally a
  dedicated road‑segmentation model for comparison.
- **Land cover:** add Sentinel‑2 ingest (reuse STAC/odc‑stac; multispectral
  bands, cloud filter). Run Prithvi‑EO via **terratorch** land‑cover head →
  class raster → colorized georeferenced PNG overlay + legend.

### Phase 5 — Model compare, evaluation, polish
- `/api/models` registry → populate Model dropdown dynamically per task.
- **Evaluation panel:** pull reference data for the AOI —
  - OSM roads (Overpass API) for roads,
  - open building footprints (Microsoft/Google Open Buildings, or OSM) for buildings,
  - ESA WorldCover for land cover —
  rasterize/align, compute **IoU / F1**, show metrics + reference overlay toggle.
- README polish: screenshots, short demo video/GIF, architecture diagram,
  write‑up of model performance findings.
- Optional: side‑by‑side two‑model compare; swipe slider; export results.

---

## 10. Open items / gotchas

- **git:** repo is **not initialized yet**. Recommended: `git init` + a clean
  commit checkpointing Phase 0–1 before the big Phase 2 install. (Was awaiting
  user go‑ahead.)
- **NAIP is US‑only.** Out‑of‑US AOIs → 422 "No NAIP imagery". (Sentinel‑2 in
  Phase 4 is global for land cover.)
- **`sys.excepthook` noise** on standalone script exit is harmless GDAL/rasterio
  teardown; does not affect results or the server.
- **AOI size cap** is 0.02°/side (`MAX_SPAN_DEG` in `stac.py`) to avoid gigapixel
  mosaics; output downsampled to ≤2048 px (`MAX_PX`). Tune for model input size.
- **Alignment check pending** (see Phase 1 note above).
- Leftover Vite template assets (`hero.png`, `icons.svg`, `vite.svg`) are unused
  but harmless; can delete during Phase 5 polish.

---

## 11. Quick resume checklist

1. `cd /home/primus/machine_learning && ./run.sh`, open http://127.0.0.1:5173
2. Draw an AOI over a US city, Fetch — confirm NAIP imagery + alignment.
3. Decide on `git init`.
4. Start Phase 2: check installed pkgs, install torch CUDA + SAM2, build
   `models/registry.py` + `models/buildings.py` + `/api/infer`, render GeoJSON.
