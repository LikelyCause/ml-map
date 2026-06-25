import { useEffect, useState } from "react";
import SplitMap from "./SplitMap";
import {
  evaluateModel,
  getHealth,
  getModels,
  getProgress,
  ingestChip,
  runInference,
  type Bbox,
  type Chip,
  type EvalResult,
  type LegendItem,
  type ModelInfo,
} from "./api";
import type { RasterOverlay } from "./SplitMap";
import "./App.css";

const TASKS = [
  { id: "buildings", label: "Building footprints" },
  { id: "roads", label: "Roads / lines of communication" },
  { id: "landcover", label: "Land cover" },
  { id: "textprompt", label: "Text-prompt segmentation" },
];

function App() {
  const [chip, setChip] = useState<Chip | null>(null);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [task, setTask] = useState(TASKS[0].id);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [source, setSource] = useState("naip");
  const [prompt, setPrompt] = useState("building");
  const [result, setResult] = useState<GeoJSON.FeatureCollection | null>(null);
  const [overlay, setOverlay] = useState<RasterOverlay | null>(null);
  const [legend, setLegend] = useState<LegendItem[] | null>(null);
  const [inferInfo, setInferInfo] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [reference, setReference] = useState<GeoJSON.FeatureCollection | null>(null);
  const [referenceOverlay, setReferenceOverlay] = useState<RasterOverlay | null>(null);
  const [referenceLegend, setReferenceLegend] = useState<LegendItem[] | null>(null);
  const [gpu, setGpu] = useState<boolean | null>(null);
  const [device, setDevice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setGpu(h.gpu);
        setDevice(h.device ?? null);
      })
      .catch(() => {
        setGpu(null);
        setDevice(null);
      });
  }, []);

  // While a long operation is in flight, poll the backend for live progress.
  const busy = loading || running || evaluating;
  useEffect(() => {
    if (!busy) {
      setStatus(null);
      return;
    }
    let active = true;
    const tick = async () => {
      const p = await getProgress();
      if (active && p && p.stage !== "idle" && p.stage !== "done") {
        setStatus(p.detail || p.stage);
      }
    };
    tick();
    const id = setInterval(tick, 400);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [busy]);

  // Load models for the selected task; reset any prior result/overlay.
  useEffect(() => {
    setResult(null);
    setOverlay(null);
    setLegend(null);
    setInferInfo(null);
    setEvalResult(null);
    setReference(null);
    setReferenceOverlay(null);
    setReferenceLegend(null);
    getModels(task).then((m) => {
      setModels(m.models);
      setModelId(m.models[0]?.id ?? "");
      setSource(m.source);
      // Different imagery source (NAIP vs Sentinel-2) => current chip is invalid.
      setChip((c) => (c && c.source && c.source !== m.source ? null : c));
    });
  }, [task]);

  const onBboxDrawn = (b: Bbox) => {
    setBbox(b);
    setDrawMode(false);
  };

  const fetchImagery = async () => {
    if (!bbox) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setOverlay(null);
    setLegend(null);
    setInferInfo(null);
    try {
      setChip(await ingestChip(bbox, source));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const runModel = async () => {
    if (!chip || !modelId) return;
    setRunning(true);
    setError(null);
    setEvalResult(null);
    setReference(null);
    setReferenceOverlay(null);
    setReferenceLegend(null);
    try {
      const res = await runInference(
        chip.id,
        task,
        modelId,
        task === "textprompt" ? prompt.trim() : undefined
      );
      if (res.overlay_url && res.bounds) {
        // Raster (land cover) result.
        setResult(null);
        setOverlay({ url: res.overlay_url, bounds: res.bounds });
        setLegend(res.legend ?? null);
        setInferInfo(`${res.legend?.length ?? 0} classes · ${res.cached ? "cached" : `${res.ms} ms`}`);
      } else {
        // Vector result.
        setOverlay(null);
        setLegend(null);
        setResult(res.geojson ?? null);
        setInferInfo(`${res.count} features · ${res.cached ? "cached" : `${res.ms} ms`} · ${modelId}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const evaluate = async () => {
    if (!chip || !modelId) return;
    setEvaluating(true);
    setError(null);
    try {
      const res = await evaluateModel(
        chip.id,
        task,
        modelId,
        task === "textprompt" ? prompt.trim() : undefined
      );
      setEvalResult(res);
      setReference(res.reference_geojson ?? null);
      setReferenceOverlay(
        res.reference_overlay_url && res.reference_bounds
          ? { url: res.reference_overlay_url, bounds: res.reference_bounds }
          : null
      );
      setReferenceLegend(res.reference_legend ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEvaluating(false);
    }
  };

  const canEval = task !== "textprompt";

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">ML‑Map</div>
        <div className="controls">
          <label>
            Task
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              {TASKS.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={models.length === 0}
            >
              {models.length === 0 ? (
                <option>— coming in a later phase —</option>
              ) : (
                models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))
              )}
            </select>
          </label>
          {task === "textprompt" && (
            <label>
              Prompt
              <input
                type="text"
                value={prompt}
                placeholder="e.g. building, parking lot, pool"
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && chip && prompt.trim() && !running) runModel();
                }}
              />
            </label>
          )}
          <button className={drawMode ? "active" : ""} onClick={() => setDrawMode((d) => !d)}>
            {drawMode ? "Drawing… (drag on left map)" : bbox ? "Redraw area" : "Select area"}
          </button>
          <button className="primary" disabled={!bbox || loading} onClick={fetchImagery}>
            {loading ? "Fetching…" : `Fetch imagery (${source === "sentinel2" ? "Sentinel-2" : "NAIP"})`}
          </button>
          <button
            className="primary"
            disabled={!chip || !modelId || running || (task === "textprompt" && !prompt.trim())}
            onClick={runModel}
          >
            {running ? "Running…" : "Run model"}
          </button>
          {canEval && (
            <button disabled={!chip || !modelId || busy} onClick={evaluate} title="Score against reference data">
              {evaluating ? "Evaluating…" : "Evaluate"}
            </button>
          )}
        </div>
        <div className="status">
          GPU: {gpu === null ? "…" : gpu ? `on${device ? ` (${device})` : ""}` : "off"}
        </div>
      </header>

      {error && <div className="error">{error}</div>}
      {busy && (
        <div className="progress">
          <span className="spinner" /> {status ?? (loading ? "Fetching imagery…" : "Running model…")}
        </div>
      )}
      {!error && !busy && (chip?.note || inferInfo) && (
        <div className="info">
          {chip?.note && (
            <span>
              {chip.note} · {chip.gsd ?? "?"} m/px · {chip.size_px?.join("×")} px
            </span>
          )}
          {inferInfo && <span className="infer"> ▸ {inferInfo}</span>}
        </div>
      )}
      <div className="stage">
        <SplitMap
          chip={chip}
          bbox={bbox}
          drawMode={drawMode}
          onBboxDrawn={onBboxDrawn}
          result={result}
          overlay={overlay}
          reference={reference}
          referenceOverlay={referenceOverlay}
        />
        {evalResult && <EvalPanel ev={evalResult} />}
        {legend && legend.length > 0 && (
          <div className="legend">
            <div className="legend-title">Land cover</div>
            {legend.map((l) => (
              <div className="legend-row" key={l.class}>
                <span className="swatch" style={{ background: l.color }} />
                <span className="legend-name">{l.class}</span>
                <span className="legend-pct">{l.pct}%</span>
              </div>
            ))}
          </div>
        )}
        {referenceLegend && referenceLegend.length > 0 && (
          <div className="legend legend-ref">
            <div className="legend-title">ESA WorldCover</div>
            {referenceLegend.map((l) => (
              <div className="legend-row" key={l.class}>
                <span className="swatch" style={{ background: l.color }} />
                <span className="legend-name">{l.class}</span>
                <span className="legend-pct">{l.pct}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EvalPanel({ ev }: { ev: EvalResult }) {
  const m = ev.metrics;
  const isVector = m.iou !== undefined && m.per_class === undefined;
  return (
    <div className="evalpanel">
      <div className="eval-title">
        Evaluation vs {ev.reference}
        {ev.ref_count !== undefined && <span className="eval-sub"> ({ev.ref_count} ref features)</span>}
      </div>
      {isVector ? (
        <div className="eval-metrics">
          <div><b>IoU</b> {m.iou}</div>
          <div><b>F1</b> {m.f1}</div>
          <div><b>Precision</b> {m.precision}</div>
          <div><b>Recall</b> {m.recall}</div>
        </div>
      ) : (
        <div className="eval-lc">
          <div className="eval-overall">Overall agreement: <b>{Math.round((m.overall_agreement ?? 0) * 100)}%</b></div>
          <table>
            <thead>
              <tr><th>class</th><th>IoU</th><th>pred%</th><th>ref%</th></tr>
            </thead>
            <tbody>
              {m.per_class?.map((c) => (
                <tr key={c.class}>
                  <td>{c.class}</td><td>{c.iou}</td><td>{c.pred_pct}</td><td>{c.ref_pct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="eval-hint">
        {isVector
          ? "cyan = reference (left) · orange = model (right)"
          : "left = ESA WorldCover · right = model classes"}
      </div>
    </div>
  );
}

export default App;
