import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Bbox, Chip } from "./api";

// Free OSM raster basemap — no API key needed for development.
const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

function boundsToImageCoords(
  b: Bbox
): [[number, number], [number, number], [number, number], [number, number]] {
  const [w, s, e, n] = b;
  return [
    [w, n],
    [e, n],
    [e, s],
    [w, s],
  ];
}

function bboxPolygon(b: Bbox): GeoJSON.Feature<GeoJSON.Polygon> {
  const [w, s, e, n] = b;
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
  };
}

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

function addOverlays(map: maplibregl.Map) {
  // AOI rectangle (both maps).
  map.addSource("aoi", { type: "geojson", data: EMPTY });
  map.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": "#38bdf8", "fill-opacity": 0.15 } });
  map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#38bdf8", "line-width": 2 } });

  // Model annotations (right/annotated map only).
  if (map.getContainer().dataset.side === "annotated") {
    map.addSource("anno", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "anno-fill", type: "fill", source: "anno", paint: { "fill-color": "#f97316", "fill-opacity": 0.4 } });
    map.addLayer({ id: "anno-line", type: "line", source: "anno", paint: { "line-color": "#fb923c", "line-width": 1.5 } });
  }

  // Reference (OSM ground truth) overlay (left/imagery map only).
  if (map.getContainer().dataset.side === "imagery") {
    map.addSource("ref", { type: "geojson", data: EMPTY });
    map.addLayer({ id: "ref-fill", type: "fill", source: "ref", paint: { "fill-color": "#22d3ee", "fill-opacity": 0.25 } });
    map.addLayer({ id: "ref-line", type: "line", source: "ref", paint: { "line-color": "#22d3ee", "line-width": 1.5 } });
  }
}

function setAoi(map: maplibregl.Map | null, bbox: Bbox | null) {
  const src = map?.getSource("aoi") as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(bbox ? { type: "FeatureCollection", features: [bboxPolygon(bbox)] } : EMPTY);
}

function setAnno(map: maplibregl.Map | null, fc: GeoJSON.FeatureCollection | null) {
  const src = map?.getSource("anno") as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(fc ?? EMPTY);
}

function setRef(map: maplibregl.Map | null, fc: GeoJSON.FeatureCollection | null) {
  const src = map?.getSource("ref") as maplibregl.GeoJSONSource | undefined;
  if (src) src.setData(fc ?? EMPTY);
}

export interface RasterOverlay {
  url: string;
  bounds: Bbox;
}

function setRasterOverlay(map: maplibregl.Map | null, ov: RasterOverlay | null) {
  if (!map) return;
  if (map.getLayer("lc")) map.removeLayer("lc");
  if (map.getSource("lc")) map.removeSource("lc");
  if (!ov) return;
  map.addSource("lc", { type: "image", url: ov.url, coordinates: boundsToImageCoords(ov.bounds) });
  // place below the AOI outline but above the imagery chip
  const before = map.getLayer("aoi-fill") ? "aoi-fill" : undefined;
  map.addLayer({ id: "lc", type: "raster", source: "lc", paint: { "raster-opacity": 0.75 } }, before);
}

function setChipLayers(map: maplibregl.Map, chip: Chip) {
  for (const id of ["chip-annotated", "chip-raw"]) {
    if (map.getLayer(id)) map.removeLayer(id);
    if (map.getSource(id)) map.removeSource(id);
  }
  const coords = boundsToImageCoords(chip.bounds);
  const annotated = map.getContainer().dataset.side === "annotated";
  map.addSource("chip-raw", { type: "image", url: chip.raw_url, coordinates: coords });
  map.addLayer({ id: "chip-raw", type: "raster", source: "chip-raw" }, "aoi-fill");
  if (annotated && chip.annotated_url) {
    map.addSource("chip-annotated", { type: "image", url: chip.annotated_url, coordinates: coords });
    map.addLayer({ id: "chip-annotated", type: "raster", source: "chip-annotated", paint: { "raster-opacity": 0.85 } }, "aoi-fill");
  }
}

interface Props {
  chip: Chip | null;
  bbox: Bbox | null;
  drawMode: boolean;
  onBboxDrawn: (b: Bbox) => void;
  result: GeoJSON.FeatureCollection | null;
  overlay: RasterOverlay | null;
  reference: GeoJSON.FeatureCollection | null;
}

export default function SplitMap({ chip, bbox, drawMode, onBboxDrawn, result, overlay, reference }: Props) {
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const leftMap = useRef<maplibregl.Map | null>(null);
  const rightMap = useRef<maplibregl.Map | null>(null);
  const drawCb = useRef(onBboxDrawn);
  drawCb.current = onBboxDrawn;

  useEffect(() => {
    if (!leftRef.current || !rightRef.current) return;
    const center: [number, number] = [-104.9932, 39.7472];
    const zoom = 15;
    leftRef.current.dataset.side = "imagery";
    rightRef.current.dataset.side = "annotated";

    const l = new maplibregl.Map({ container: leftRef.current, style: BASE_STYLE, center, zoom });
    const r = new maplibregl.Map({ container: rightRef.current, style: BASE_STYLE, center, zoom });
    leftMap.current = l;
    rightMap.current = r;

    let syncing = false;
    const sync = (from: maplibregl.Map, to: maplibregl.Map) => () => {
      if (syncing) return;
      syncing = true;
      to.jumpTo({ center: from.getCenter(), zoom: from.getZoom(), bearing: from.getBearing(), pitch: from.getPitch() });
      syncing = false;
    };
    l.on("move", sync(l, r));
    r.on("move", sync(r, l));
    l.on("load", () => addOverlays(l));
    r.on("load", () => addOverlays(r));

    return () => {
      l.remove();
      r.remove();
    };
  }, []);

  // AOI drawing (drag on the left map) — attach only while in draw mode.
  useEffect(() => {
    const map = leftMap.current;
    if (!map || !drawMode) return;
    map.getCanvas().style.cursor = "crosshair";
    map.dragPan.disable();
    let start: maplibregl.LngLat | null = null;

    const onDown = (e: maplibregl.MapMouseEvent) => {
      start = e.lngLat;
    };
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (!start) return;
      const b: Bbox = [
        Math.min(start.lng, e.lngLat.lng),
        Math.min(start.lat, e.lngLat.lat),
        Math.max(start.lng, e.lngLat.lng),
        Math.max(start.lat, e.lngLat.lat),
      ];
      setAoi(leftMap.current, b);
      setAoi(rightMap.current, b);
    };
    const onUp = (e: maplibregl.MapMouseEvent) => {
      if (!start) return;
      const b: Bbox = [
        Math.min(start.lng, e.lngLat.lng),
        Math.min(start.lat, e.lngLat.lat),
        Math.max(start.lng, e.lngLat.lng),
        Math.max(start.lat, e.lngLat.lat),
      ];
      start = null;
      if (b[2] > b[0] && b[3] > b[1]) drawCb.current(b);
    };

    map.on("mousedown", onDown);
    map.on("mousemove", onMove);
    map.on("mouseup", onUp);
    return () => {
      map.off("mousedown", onDown);
      map.off("mousemove", onMove);
      map.off("mouseup", onUp);
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
    };
  }, [drawMode]);

  // Keep the AOI rectangle in sync with state (e.g. after drawing finishes).
  useEffect(() => {
    setAoi(leftMap.current, bbox);
    setAoi(rightMap.current, bbox);
  }, [bbox]);

  // Update model-annotation overlay on the right map.
  useEffect(() => {
    const map = rightMap.current;
    if (!map) return;
    if (map.isStyleLoaded() && map.getSource("anno")) setAnno(map, result);
    else map.once("idle", () => setAnno(map, result));
  }, [result]);

  // Update land-cover raster overlay on the right map.
  useEffect(() => {
    const map = rightMap.current;
    if (!map) return;
    if (map.isStyleLoaded()) setRasterOverlay(map, overlay);
    else map.once("idle", () => setRasterOverlay(map, overlay));
  }, [overlay]);

  // Update reference (ground-truth) overlay on the left map.
  useEffect(() => {
    const map = leftMap.current;
    if (!map) return;
    if (map.isStyleLoaded() && map.getSource("ref")) setRef(map, reference);
    else map.once("idle", () => setRef(map, reference));
  }, [reference]);

  // (Re)load chip overlays whenever the chip changes.
  useEffect(() => {
    if (!chip) return;
    const apply = (map: maplibregl.Map | null) => {
      if (!map) return;
      if (map.isStyleLoaded() && map.getLayer("aoi-fill")) setChipLayers(map, chip);
      else map.once("idle", () => setChipLayers(map, chip));
    };
    apply(leftMap.current);
    apply(rightMap.current);
  }, [chip]);

  return (
    <div className="split">
      <div className="pane">
        <span className="pane-label">Imagery</span>
        <div ref={leftRef} className="map" />
      </div>
      <div className="pane">
        <span className="pane-label">Annotated</span>
        <div ref={rightRef} className="map" />
      </div>
    </div>
  );
}
