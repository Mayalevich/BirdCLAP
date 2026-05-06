from __future__ import annotations

import mimetypes
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.config import load_settings
from backend.errors import BackendError
from backend.provider_factory import build_provider
from backend.schemas import (
    ClassificationResponse,
    DescribeAudioResponse,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
)

_settings = load_settings()
_provider = None
_provider_lock = threading.Lock()

# ── Audio map built at startup — independent of the CLAP model ───────────────
# Serves /api/audio/{recording_id} without ever loading the model.
_audio_map: dict[str, Path] = {}


def _build_startup_audio_map() -> dict[str, Path]:
    """Scan audio_root by file stem.  No model required.  Runs once at startup."""
    root = Path(_settings.audio_root)
    if not root.is_dir():
        return {}
    stem_map: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in (".wav", ".mp3", ".ogg", ".flac"):
            continue
        stem = path.stem
        existing = stem_map.get(stem)
        # Prefer MP3 (full original recording) over WAV (10-second training sidecar)
        if existing is None:
            stem_map[stem] = path
        elif path.suffix.lower() == ".mp3" and existing.suffix.lower() != ".mp3":
            stem_map[stem] = path
    return stem_map


def _get_provider():
    """Load GPU/model only after uvicorn binds the port (avoids long warmup then bind failure)."""
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = build_provider(_settings)
    return _provider


app = FastAPI(title="Lets Solve It Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global _audio_map
    _audio_map = _build_startup_audio_map()


@app.exception_handler(BackendError)
def handle_backend_error(_request, exc: BackendError):
    payload = ErrorResponse(code=exc.code, message=exc.message).model_dump()
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "provider": _settings.model_provider,
        "model_ready": str(_provider is not None).lower(),
        "audio_files_indexed": str(len(_audio_map)),
    }


@app.post(
    "/api/search",
    response_model=SearchResponse,
    responses={501: {"model": ErrorResponse}},
)
def search(body: SearchRequest):
    results = _get_provider().search_text(query=body.query.strip(), top_k=body.top_k)
    return SearchResponse(query=body.query.strip(), count=len(results), results=results)


@app.post(
    "/api/search-by-audio",
    response_model=SearchResponse,
    responses={501: {"model": ErrorResponse}},
)
async def search_by_audio(file: UploadFile = File(...), top_k: int = 10):
    content = await file.read()
    results = _get_provider().search_by_audio(
        file_bytes=content, filename=file.filename or "upload.wav", top_k=top_k
    )
    return SearchResponse(query=file.filename or "uploaded-audio", count=len(results), results=results)


@app.post(
    "/api/classify-audio",
    response_model=ClassificationResponse,
    responses={501: {"model": ErrorResponse}},
)
async def classify_audio(file: UploadFile = File(...), top_k: int = 5):
    content = await file.read()
    hits = _get_provider().classify_audio(
        file_bytes=content, filename=file.filename or "upload.wav", top_k=top_k
    )
    return ClassificationResponse(count=len(hits), results=hits)


@app.get(
    "/api/audio/{recording_id}",
    summary="Stream a gallery recording audio file — no model load required",
)
def serve_audio(recording_id: str):
    # Use the startup-built map: pure file serving, never waits for the CLAP model.
    path = _audio_map.get(recording_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio not found: {recording_id!r}")
    media_type = mimetypes.guess_type(str(path))[0] or "audio/octet-stream"
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@app.post(
    "/api/describe-audio",
    response_model=DescribeAudioResponse,
    responses={501: {"model": ErrorResponse}},
    summary="Return acoustic text descriptions nearest to the uploaded audio embedding",
)
async def describe_audio(file: UploadFile = File(...), top_k: int = 4):
    content = await file.read()
    descriptions = _get_provider().describe_audio(
        file_bytes=content, filename=file.filename or "upload.wav", top_k=top_k
    )
    return DescribeAudioResponse(descriptions=descriptions)
