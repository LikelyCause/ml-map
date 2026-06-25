# Swath — frontend

React + TypeScript + MapLibre GL (Vite). The split-map UI: draw an AOI, fetch
imagery, run a model, and compare results side by side.

```bash
npm install
npm run dev   # http://127.0.0.1:5173 (proxies /api and /data to the backend on :8077)
```

Usually you don't run this directly — `../run.sh` starts the backend and this dev
server together. See the root [README](../README.md).
