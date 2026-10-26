"""FastAPI backend for the Ripple dashboard.

Serves replay data precomputed by ``ripple demo`` (no model code in the server) and an optional
LLM briefing for the operations desk. Run: ``uv run ripple serve``.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ripple import config
from ripple.api.briefing import briefing

log = logging.getLogger(__name__)
DATASET = "synthetic"
WEB_DIST = config.ROOT / "app" / "web" / "dist"

app = FastAPI(title="Ripple", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"],
                   allow_headers=["*"])  # fmt: skip


def _demo_dir() -> Path:
    return config.ARTIFACTS / DATASET / "demo"


@lru_cache(maxsize=1)
def _index() -> dict:
    p = _demo_dir() / "index.json"
    if not p.exists():
        raise HTTPException(503, "No demo data. Run `uv run ripple demo` first.")
    return json.loads(p.read_text())


@lru_cache(maxsize=16)
def _day(day: str, model: str) -> dict:
    p = _demo_dir() / f"{day}_{model}.json"
    if not p.exists():
        raise HTTPException(404, f"no replay for {day} / {model}")
    return json.loads(p.read_text())


@app.get("/api/index")
def index() -> dict:
    return _index()


@app.get("/api/day/{day}/{model}")
def day(day: str, model: str) -> dict:
    if day not in _index()["days"] or model not in _index()["models"]:
        raise HTTPException(404, "unknown day or model")
    return _day(day, model)


class BriefingRequest(BaseModel):
    day: str
    model: str
    frame: int


@app.post("/api/briefing")
def make_briefing(req: BriefingRequest) -> dict:
    d = _day(req.day, req.model)
    if not 0 <= req.frame < len(d["frames"]):
        raise HTTPException(400, "frame out of range")
    return briefing(d, req.frame, synthetic=_index()["dataset"] == "synthetic")


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
