from __future__ import annotations

import threading

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import load_settings
from backend.errors import BackendError
from backend.provider_factory import build_provider
from backend.schemas import ClassificationResponse, ErrorResponse, SearchRequest, SearchResponse

_settings = load_settings()
_provider = None
_provider_lock = threading.Lock()


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
    results = _get_provider().search_by_audio(file_bytes=content, filename=file.filename or "upload.wav", top_k=top_k)
    return SearchResponse(query=file.filename or "uploaded-audio", count=len(results), results=results)


@app.post(
    "/api/classify-audio",
    response_model=ClassificationResponse,
    responses={501: {"model": ErrorResponse}},
)
async def classify_audio(file: UploadFile = File(...), top_k: int = 5):
    content = await file.read()
    hits = _get_provider().classify_audio(file_bytes=content, filename=file.filename or "upload.wav", top_k=top_k)
    return ClassificationResponse(count=len(hits), results=hits)
