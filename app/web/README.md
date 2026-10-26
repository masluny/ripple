# Ripple dashboard

React + TypeScript (Vite) front end for the Ripple replay API.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api to http://127.0.0.1:8000 (uv run ripple serve)
npm run build    # dist/ is served by `uv run ripple serve` at http://127.0.0.1:8000
```

Map: azimuthal equidistant projection centred on WAW with a compressed radius (d3-geo), so every
route is a straight spoke. Land: Natural Earth via `world-atlas`, bundled with the app (no tile server).
