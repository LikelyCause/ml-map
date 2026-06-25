// Backend API client. Dev requests are proxied to FastAPI by Vite (see vite.config.ts).

export interface Chip {
  id: string;
  bounds: [number, number, number, number]; // [west, south, east, north]
  raw_url: string;
  annotated_url: string | null;
  source?: string;
  datetime?: string;
  gsd?: number;
  size_px?: [number, number];
  note?: string;
}

export async function getDemoChip(): Promise<Chip> {
  const res = await fetch("/api/demo");
  if (!res.ok) throw new Error(`demo chip request failed: ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<{ status: string; gpu: boolean; device?: string }> {
  const res = await fetch("/api/health");
  return res.json();
}

export interface Progress {
  stage: string;
  detail: string;
  pct: number | null;
}

export async function getProgress(): Promise<Progress | null> {
  try {
    const res = await fetch("/api/progress");
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export interface ModelInfo {
  id: string;
  hf: string;
  label: string;
}

export interface TaskModels {
  models: ModelInfo[];
  source: string; // "naip" or "sentinel2"
}

export async function getModels(task: string): Promise<TaskModels> {
  const res = await fetch(`/api/models?task=${encodeURIComponent(task)}`);
  if (!res.ok) return { models: [], source: "naip" };
  const d = await res.json();
  return { models: d.models ?? [], source: d.source ?? "naip" };
}

export interface LegendItem {
  class: string;
  color: string;
  pct: number;
}

export interface InferResult {
  task: string;
  model_id: string;
  cached: boolean;
  count?: number;
  ms?: number;
  geojson?: GeoJSON.FeatureCollection;
  // land cover (raster) results:
  overlay_url?: string;
  bounds?: [number, number, number, number];
  legend?: LegendItem[];
}

export async function runInference(
  chip_id: string,
  task: string,
  model_id: string,
  prompt?: string
): Promise<InferResult> {
  const res = await fetch("/api/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chip_id, task, model_id, prompt }),
  });
  if (!res.ok) {
    let detail = `inference failed: ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json();
}

export interface EvalResult {
  task: string;
  model_id: string;
  reference: string;
  // vector metrics OR land-cover agreement (shape varies by task)
  metrics: Record<string, number> & {
    per_class?: { class: string; iou: number; pred_pct: number; ref_pct: number }[];
    overall_agreement?: number;
  };
  reference_geojson?: GeoJSON.FeatureCollection;
  ref_count?: number;
  // land-cover reference: colorized ESA WorldCover raster overlay (left pane)
  reference_overlay_url?: string;
  reference_bounds?: [number, number, number, number];
  reference_legend?: LegendItem[];
}

export async function evaluateModel(
  chip_id: string,
  task: string,
  model_id: string,
  prompt?: string
): Promise<EvalResult> {
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chip_id, task, model_id, prompt }),
  });
  if (!res.ok) {
    let detail = `evaluation failed: ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json();
}

export type Bbox = [number, number, number, number]; // [west, south, east, north]

export async function ingestChip(bbox: Bbox, source = "naip"): Promise<Chip> {
  const res = await fetch("/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bbox, source }),
  });
  if (!res.ok) {
    let detail = `ingest failed: ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json();
}
