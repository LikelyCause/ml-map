# Swath — Geospatial Foundation Models, Side by Side

An interactive app for applying **geospatial foundation models** to remotely
sensed imagery — **inference only, no training**. Draw an area on the map,
one‑click fetch high‑resolution imagery, run a foundation model, and compare its
output against the raw image **and against real reference data** with hard
metrics.

The point isn't that one model does everything — it's the honest comparison:
**different foundation models excel at different tasks, and some tasks are still
hard zero‑shot.** The app makes that visible and quantifiable.

## Why I built this

I was curious how well both geospatial foundation models and general CV models actually perform on satellite imagery for basic tasks, and decided to build a tool to see them in action.

Swath: not my best naming work, but a fun app!

1) Draw a box on a map
2) One-click pull the imagery (NAIP / Sentinel-2; reprojects/mosaicks for you!)
3) Choose your model from a dropdown (G-DINO+SAM, SAM Everything+ViT Base-Huge, Clay v1 Base)
4) Run model and compare imagery with annotations side by side
5) Score (some of) it against real reference data (OSM, ESA WorldCover)

To no one's surprise, no single model does everything zero-shot. Clean buildings can be extracted pretty well with DINO+SAM, but anything dense and you get massive blobs (even an NDVI vegetation filter only helps so much) - and forget about extracting roads once there's any foliage.

I added some validation metrics out of curiosity to measure *how* bad things were doing, and... well, bad. Good thing this isn't for production! This project makes me really curious how well these models could be tuned using open data sources, like SpaceNet.

### Where it goes next

- **✓ More Prithvi‑EO heads** *(done)* — fine‑tuned Prithvi‑EO‑2.0 heads for
  burn‑scar and flood, on the same encoder.
- **✓ Clay as an embeddings demo** *(done)* — Clay and Prithvi‑EO‑2.0 encoders run
  unsupervised, showing what a foundation model actually outputs (embeddings, not
  labels) clustered into self‑similar regions.
- **Train task‑specific specialists** — the research kept pointing to the same
  answer for the hard cases: zero‑shot foundation models give you blobs; dense
  building footprints and connected road networks need a *trained* specialist
  (e.g. a polygon/segmentation head on **SpaceNet** / Inria / DeepGlobe). Proving
  that — and then building it — is half the value. See
  [docs/building-extraction-research.md](docs/building-extraction-research.md)
  for the cited verdict.

> Along the way I added some non‑foundation models too — handy for comparison, if
> slightly off the central theme. The roadmap above steers it back toward the
> foundation‑model thesis.

Example outputs (raw imagery | model annotation):

| Building footprints (SAM Everything + ViT-Huge, NAIP) | Land cover (Prithvi-EO 100M vs ESA WorldCover, Sentinel-2) |
|---|---|
| ![buildings](docs/building_footprints_small.png) | ![land cover](docs/land_cover.png) |

---

## What it does

- **Split view** — raw imagery (left) vs. model annotations (right), with synced pan/zoom.
- **Seven tasks**, each with a model dropdown to compare checkpoints:
  1. **Building footprints** — SAM "segment‑everything" *vs.* Grounding DINO + SAM (with shape + NDVI vegetation filters)
  2. **Roads / lines of communication** — Grounding DINO + SAM, linearity‑filtered
  3. **Land cover** — Prithvi‑EO‑1.0 (13‑class crop/land), multi‑temporal Sentinel‑2
  4. **Foundation embeddings (unsupervised)** — run a **Clay v1** or **Prithvi‑EO‑2.0** encoder, k‑means the patch embeddings into self‑similar regions (zero‑shot, no task head)
  5. **Burn scars (wildfire)** — **Prithvi‑EO‑2.0‑300M** fine‑tuned (HLS Burn Scars)
  6. **Flood / surface water** — **Prithvi‑EO‑2.0‑300M** fine‑tuned (Sen1Floods11)
  7. **Text‑prompt segmentation** — open vocabulary: type any object and segment it
- **One‑click AOI ingest** — draw a box → fetch NAIP (0.3 m) or Sentinel‑2 from the Microsoft Planetary Computer.
- **Live progress** — granular status (catalog search → tile download → model load → tiled detect → mask → vectorize).
- **Evaluation** — score predictions against free reference data (OpenStreetMap, ESA WorldCover) with **IoU / precision / recall / F1**, plus a side‑by‑side reference overlay: cyan ground‑truth vectors for buildings/roads, and a colorized **ESA WorldCover** raster (left) vs. the model's classes (right) for land cover.

## Results (zero‑shot, example AOIs)

| Task | Model | Reference | Metric |
|------|-------|-----------|--------|
| Buildings | **Grounding DINO + SAM** | OSM footprints | **IoU 0.53 · F1 0.69 · P 0.82 · R 0.60** |
| Buildings | SAM segment‑everything | OSM footprints | IoU 0.41 · F1 0.58 |
| Roads | Grounding DINO + SAM | OSM roads | P 0.66 · **R 0.21** · F1 0.32 |
| Land cover | Prithvi‑EO‑1.0 100M | ESA WorldCover | overall agreement varies by AOI |

**Reading the numbers:**
- **Buildings:** on clean, well‑separated scenes detect‑then‑segment (DINO+SAM) edges out segment‑everything — but over **dense clusters** DINO+SAM collapses to block‑scale blobs, and **SAM segment‑everything** (with shape + NDVI vegetation filters) is the better default. The honest fix is a polygon specialist — see [docs/building-extraction-research.md](docs/building-extraction-research.md).
- **Roads:** high precision, low recall — the model finds prominent arterials and rail corridors but misses the residential street grid. Zero‑shot foundation models can't do full road extraction; you'd want a road‑specific model (SpaceNet/DeepGlobe).
- **Land cover:** the CDL‑trained crop model works well over Midwest farmland (sensible corn/soy/wheat) but over‑predicts "cropland" elsewhere — specialist models don't generalize.
- **Foundation embeddings / burn‑scar / flood:** qualitative (no fixed reference). The embeddings task shows *what an encoder considers similar* zero‑shot; the fine‑tuned Prithvi‑2.0 heads are supervised, so they produce real burn‑scar / water masks where those features are present.

## Data & models

| | Source | Resolution | Access |
|---|---|---|---|
| High‑res imagery | **NAIP** (US) | 0.3–1 m | Planetary Computer STAC |
| Multispectral | **Sentinel‑2 L2A** (global) | 10–20 m | Planetary Computer STAC |
| Buildings/roads/text | **SAM**, **Grounding DINO** | — | HuggingFace `transformers` |
| Land cover | **Prithvi‑EO‑1.0‑100M** (crop classification) | — | HuggingFace (mmseg checkpoint, rebuilt in PyTorch) |
| Foundation encoders | **Clay v1**, **Prithvi‑EO‑2.0‑300M** | — | `terratorch` backbone registry |
| Fine‑tuned heads | **Prithvi‑EO‑2.0‑300M** BurnScars · Sen1Floods11 | — | HuggingFace (native terratorch checkpoints, Apache‑2.0) |
| Reference | **OpenStreetMap** (Overpass), **ESA WorldCover** | — | Overpass API / Planetary Computer |

## Architecture

```
React + MapLibre GL (Vite, :5173)            FastAPI (:8077, CUDA / Apple MPS)
 ├─ AOI draw → /ingest                  →      ├─ /ingest   NAIP / Sentinel-2 (STAC mosaic, reproject)
 ├─ task ▸ model ▸ prompt → /infer      →      ├─ /infer    SAM · DINO+SAM (tiled) · Prithvi land cover
 │                                             │             · GeoFM embeddings (Clay/Prithvi-2.0 → k-means)
 │                                             │             · fine-tuned Prithvi-2.0 (burn scar / flood)
 ├─ Evaluate → /evaluate                →      ├─ /evaluate OSM / WorldCover → IoU/F1 (+ WorldCover overlay)
 ├─ split maps + overlays               ←      ├─ /progress live stage tracker
 └─ progress banner (polls /progress)         └─ model zoo (lazy-loaded, kept warm in GPU / unified memory)
```

Chips are small per‑AOI, so results are served as MapLibre `ImageSource` raster
overlays (PNG + bounds) and GeoJSON vectors — no tile server needed. All the
Sentinel‑2 tasks (land cover, embeddings, burn scar, flood) reuse one 6‑band
ingest, so adding a model is a registry entry + a wrapper.

## Notable engineering

- **Ran a legacy model on a modern stack.** The Prithvi land‑cover model ships
  only as an mmsegmentation checkpoint (won't run under torch 2.6). Dissected the
  `.pth` and **rebuilt the exact architecture in plain PyTorch** — the legacy
  weights load with zero missing/unexpected keys.
- **Runs natively on Apple Silicon (MPS).** A central accelerator selector picks
  cuda → mps → cpu, and individual MPS op gaps fall back to CPU per‑op rather than
  crashing: Grounding DINO's MPSGraph shape assertion (detector pinned to CPU),
  SAM's float64 box tensors (down‑cast), and the flood model's UperNet
  adaptive‑pool (model relocated to CPU on first failure).
- **Geospatial foundation models via terratorch.** Clay v1 and Prithvi‑EO‑2.0
  encoders produce per‑patch embeddings that are k‑means clustered into an
  unsupervised segmentation; two released Prithvi‑EO‑2.0 fine‑tunes (burn scar,
  flood) load as native terratorch checkpoints. All consume the existing 6‑band
  Sentinel‑2 stack.
- **Tiled (SAHI‑style) detection** at full native resolution so Grounding DINO
  (trained on ground‑level photos) finds far more overhead objects (~3× recall).
- **Honest, model‑quality fixes** (auditable in [docs/building-extraction-research.md](docs/building-extraction-research.md)):
  a roads linearity filter and building footprint/NDVI‑vegetation filters; an
  EPSG:4326 plate‑carrée **de‑stretch** so the detector sees ground‑square pixels;
  and an eval‑correctness pass (only the line‑geometry road reference is buffered).
- **Sentinel‑2 gotcha:** removed the +1000 baseline‑04.00 BOA offset to match the
  reflectance units Prithvi was trained on (without it, every class was wrong).
- **NAIP mosaic** spans all overlapping tiles, newest‑first, so an AOI that
  straddles multiple capture dates still fills the whole box.

## Stack

- **Backend:** FastAPI · PyTorch 2.6 · transformers · **terratorch** (Prithvi /
  Clay / fine‑tuned heads) · rasterio/rioxarray/odc‑stac · shapely/geopandas.
  Python 3.12 (pyenv venv).
- **Frontend:** React + TypeScript + MapLibre GL (Vite).
- **Hardware:** developed on an RTX 4080 Super (16 GB) and on an Apple Silicon
  Mac (M‑series, MPS). The accelerator is auto‑selected at runtime
  (cuda → mps → cpu), so there are no per‑machine code changes; all inference is
  local. On Apple Silicon install the default PyPI torch wheels (no CUDA index)
  and `run.sh` exports `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Setup

Requires **Python 3.12** and **Node 18+**.

```bash
# 1) Backend — create a venv and install deps. Install torch FIRST; see the
#    backend/requirements.txt header for the NVIDIA-vs-Apple-Silicon command.
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 2) Frontend
cd frontend && npm install && cd ..
```

Model weights download on first use (the progress banner shows them).

## Run (development)

```bash
./run.sh
# backend  → http://127.0.0.1:8077  (FastAPI, docs at /docs)
# frontend → http://127.0.0.1:5173  (open this)
```

The frontend proxies `/api` and `/data` to the backend, so just open the frontend URL.

**Tips:**
- NAIP is US‑only (≤ ~5.5 km AOI; `SWATH_NAIP_MAX_SPAN_DEG` to raise). Sentinel‑2
  tasks allow ~20 km.
- Land cover is US‑cropland‑trained — try farmland (e.g. central Iowa) for
  sensible results; burn‑scar/flood want scenes that actually contain fire/water.
- First run of each model downloads its weights — the progress banner shows it
  (Prithvi‑EO‑2.0‑300M ≈ 1.2–1.3 GB; SAM/DINO smaller).

## Build status

- [x] **Phase 0** — backend skeleton + React/MapLibre split view
- [x] **Phase 1** — AOI draw → STAC ingest (NAIP, 0.3 m)
- [x] **Phase 2** — Building footprints (SAM segment‑everything) + model dropdown
- [x] **Phase 3** — Text‑prompt segmentation (Grounding DINO + SAM, tiled)
- [x] **Phase 4** — Roads + Land cover (Sentinel‑2 + Prithvi 13‑class) + live progress UI + full‑res tiling
- [x] **Phase 5** — Model comparison + evaluation vs OSM / ESA WorldCover (IoU/F1) + reference overlay
- [x] **Phase 6** — Apple Silicon (MPS) support; model‑quality pass (eval audit, roads linearity + building NDVI/shape filters, imagery de‑stretch); colorized WorldCover overlay; NAIP multi‑date mosaic
- [x] **Phase 7** — Geospatial foundation models: Clay + Prithvi‑EO‑2.0 embeddings (unsupervised) and fine‑tuned Prithvi‑EO‑2.0 burn‑scar / flood

## Limitations (by design — this is a zero‑shot study)

- NAIP is US‑only; the crop/land model is US‑cropland‑trained.
- Roads are not solved zero‑shot (low recall); linear features under text‑prompt
  (e.g. "shoreline") hit the same `detect→box→SAM` ceiling.
- SAM mask boundaries are limited by its internal 1024 px encoder.
- **Dense, tree‑covered suburbia is hard:** SAM masks tree canopy as buildings
  (high recall, low precision). An **NDVI vegetation filter** (using NAIP's NIR
  band) plus shape filters mitigate it; a building specialist would solve it.
- **Foundation embeddings are unsupervised** — clusters are self‑similar regions,
  not named classes; normalization is approximate across encoders.
- Reference data is imperfect (OSM completeness varies; WorldCover is 10 m and a
  different taxonomy than the crop model) — metrics are indicative, not absolute.

![dense buildings](docs/building_footprints.png)

_SAM "segment‑everything" over dense, foliage‑covered suburbia: it finds the
houses (high recall) but mis‑segments tree canopy as buildings — the precision
gap the NDVI filter targets. See [docs/building-extraction-research.md](docs/building-extraction-research.md)._

## License

[MIT](LICENSE) © Tanner Overcash
