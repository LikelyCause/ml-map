# Building & road extraction — research notes and model-quality fixes

This document captures (1) a cited research evaluation of the *right tool* for
dense building footprint extraction, and (2) the practical model-quality fixes
applied to Swath's zero-shot pipeline along the way. It exists because the
zero-shot SAM / Grounding-DINO+SAM building path produces **block-scale blobs**
over clustered houses and **masks tree canopy as buildings** over foliage, and
the roads path can't see a residential street grid at all.

---

## Part 1 — The right tool for dense building footprints (researched verdict)

Question: what should replace zero-shot SAM/DINO+SAM for extracting dense
building footprints from US NAIP (0.3–1 m RGB+NIR), preferring off-the-shelf
weights and clean vector polygons?

### (a) Fastest accurate path — *fetch* precomputed footprints
- **Microsoft US Building Footprints** (~129.6M polygons, all 50 states + DC,
  free GeoJSON) · **Overture** / **VIDA** GeoParquet fusions (read natively by
  geopandas). <https://github.com/microsoft/USBuildingFootprints> ·
  <https://docs.overturemaps.org/guides/buildings/> ·
  <https://source.coop/vida/google-microsoft-osm-open-buildings>
- **Two catches:** (1) all **ODbL** (share-alike — derivatives must be
  relicensed); (2) they were made by the *same* CNN-segmentation→polygonize
  pipeline we're trying to escape, and a peer-reviewed ORNL study shows they
  **also fail dense instance separation** — Microsoft merges adjacent houses
  into one polygon, Google over-segments. <https://www.osti.gov/servlets/purl/2000384>
- **Verdict:** best as a *reference/baseline layer* (better rural coverage than
  OSM), not a cure for the dense case.

### (b) Best model to actually RUN on NAIP — direct polygon models
- **Pix2Poly (recommended)** — MIT license, pretrained checkpoints for
  **Inria / SpaceNet2 / WHU / Massachusetts** (Inria is 0.3 m aerial — a strong
  NAIP match). Outputs explicit **ring-graph polygons** directly (no masks).
  Beats HiSup and Frame Field Learning on Inria vector metrics
  (C-IoU 71.73 vs 66.1 / 49.8; PoLiS 1.914 vs 2.438 / 2.865).
  <https://github.com/yeshwanth95/Pix2Poly>
  - ⚠️ License gotcha: the *bundled* P3 distribution (HF `rsi/PixelsPointsPolygons`)
    is **academic-non-commercial**. Use Pix2Poly's **standalone MIT repo**
    checkpoints instead. <https://github.com/raphaelsulzer/PixelsPointsPolygons>
- **HiSup** — HRNetV2-W48 weights (AICrowd + Inria), direct polygons; but an
  independent report notes real-world IoU below paper numbers and degradation on
  densely-spaced buildings. <https://github.com/SarahwXU/HiSup>
- **Frame Field Learning** — ready-to-run, but 2020 weights predating NAIP tuning.
  <https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning>
- **TernausNetV2** — purpose-built for *touching* buildings (object-boundary mask
  + watershed), but outputs raster (needs vectorizing) and expects 11 bands
  (needs input-layer surgery for NAIP's 4). <https://github.com/ternaus/TernausNetV2>

### (c) The "geospatial foundation model" theme — no free lunch
No ready-to-run building-footprint head was found for Prithvi-EO/terratorch,
Clay, SatMAE/Scale-MAE, DOFA, or SAM-for-RS. Staying on-theme means
**fine-tuning your own head** on Inria/SpaceNet — a *training* effort, outside an
inference-only budget.

### Takeaway
The honest best tool for buildings is a **task-specific specialist (Pix2Poly)**,
not a foundation model — which *strengthens* the app's thesis: zero-shot
foundation models produce blobs; some tasks need a trained specialist.

> Roads are the analogous-but-harder case: zero-shot detect→box→SAM can't trace a
> connected network (recall ~0.03 on a tree-occluded grid). The right tool is a
> road **semantic-segmentation** specialist (SpaceNet Roads, DeepGlobe,
> Massachusetts Roads, or **CRESI** for a routable centerline graph), then
> skeletonize. In well-mapped areas, OSM already has the roads — "fetch, don't
> extract."

---

## Part 2 — Model-quality fixes applied this session

These are honest, in-stack improvements to the zero-shot pipeline (they raise the
floor; they don't replace a specialist model).

### Evaluation correctness
- **Roads double-buffer bug (fixed).** Road predictions are already areal SAM
  polygons, but `evaluate.py` buffered them by the same +6 m as the line-geometry
  OSM reference, inflating predicted area ~12% (up to 2× on thin chips) and
  unfairly deflating road precision/IoU. Now only the reference is buffered.
  (`backend/eval/evaluate.py`)
- **Verified non-bugs:** the projection pipeline does **not** corrupt metrics
  (pred and reference rasterize on the same grid; proven IoU-invariant across a
  degree grid vs a conformal UTM grid), `from_bounds` arg order, north-up
  orientation, OSM bbox order, and the IoU/P/R/F1 math are all correct.

### Shape filters (`backend/models/textprompt.py`)
- **Roads linearity filter** — box-prompted SAM otherwise masks the dominant
  region in each box (a field/lawn parcel). Drop masks with ground-meter
  elongation < 4. Measured separation: real roads ≳13, field blobs ≲3.
- **Buildings footprint filters** — the DINO+SAM buildings path now reuses the
  segment-everything area/extent/aspect/image-fraction filters, so it stops
  emitting block-scale blobs.

### NDVI vegetation filter (the foliage fix) — `backend/ingest/stac.py`, `backend/models/buildings.py`
Over tree-heavy suburbia, SAM segment-everything finds the buildings
(recall ~0.80) but masks **tree canopy** as buildings too (precision ~0.29), and
geometric filters can't tell a round tree from a roof. The discriminating signal
was already in the data and being discarded: **NAIP band 4 is NIR**.
- Ingest now keeps NIR and saves an NDVI sidecar (`{chip}_ndvi.npy`),
  uint8-encoded and de-stretched to the PNG layout.
- Both building paths reject masks whose **mean NDVI > 0.3** (vegetation:
  trees/lawns ≈ 0.4–0.8; roofs/pavement < 0.15). Buildings-only; land-cover,
  roads, and free-text prompts are untouched.
- Expected effect on the foliage scene: precision 0.29 → ~0.5–0.7 with recall
  barely moving (roofs kept, trees dropped). Tunable via `NDVI_VEG_THRESH`.

### Imagery de-stretch (`backend/ingest/stac.py`)
NAIP chips were reprojected to EPSG:4326 (plate carrée), stretching imagery ~31%
horizontally at 40°N — distorting the object aspect ratios DINO/SAM rely on.
Chips are now compressed to ground-square pixels (georeferencing and display
unaffected; verified IoU-invariant).

> Both the NDVI sidecar and the de-stretch only affect **newly fetched** chips —
> re-fetch an AOI to benefit. `PIPE_VERSION` is bumped so cached inference re-runs.

---

## Research provenance
Part 1 is a multi-source, adversarially-verified deep-research run (2026-06-24);
Part 2 reflects an eval/projection audit plus the implemented fixes. See repo
memory `dense-building-footprint-tool.md` for the condensed verdict.
