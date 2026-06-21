"""FastAPI app: serves the prediction API and the built React frontend from a
single origin (one container, one port - easy to deploy on Render).

Data tabs (options/areas/map/metrics) are served from precomputed JSON, so the
heavy ML service is loaded lazily on the first /api/predict call only.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import PredictRequest

ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
FIGURES_DIR = ROOT / "reports" / "figures"
FRONTEND_DIST = ROOT / "app" / "frontend" / "dist"

app = FastAPI(title="Gridlock Traffic Intelligence", version="1.0.0")

# Allow the Vite dev server (localhost:5173) to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_JSON_CACHE: dict[str, object] = {}


def _serve_generated(name: str):
    if name in _JSON_CACHE:
        return _JSON_CACHE[name]
    path = GENERATED_DIR / name
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{name} not found. Run `python -m app.backend.precompute` to generate it.",
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    _JSON_CACHE[name] = data
    return data


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/options")
def options():
    return _serve_generated("options.json")


@app.get("/api/areas")
def areas():
    return _serve_generated("areas.json")


@app.get("/api/map")
def map_data():
    return _serve_generated("map.json")


@app.get("/api/metrics")
def metrics():
    return _serve_generated("metrics.json")


@app.post("/api/predict")
def predict(req: PredictRequest):
    from .inference import get_service

    try:
        service = get_service()
        return service.predict(req.to_payload())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/api/figures/{name}")
def figure(name: str):
    # Prevent path traversal; only serve known PNGs from the figures folder.
    safe = Path(name).name
    path = FIGURES_DIR / safe
    if path.suffix.lower() != ".png" or not path.exists():
        raise HTTPException(status_code=404, detail="figure not found")
    return FileResponse(path, media_type="image/png")


# --------------------------------------------------------------------------- #
# Static frontend (single-origin). Mounted last so /api/* always wins.
# --------------------------------------------------------------------------- #
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:  # pragma: no cover - dev mode before the frontend is built
    @app.get("/")
    def index_dev():
        return JSONResponse(
            {"message": "Gridlock API running. Build the frontend (npm run build) or use the Vite dev server."}
        )
