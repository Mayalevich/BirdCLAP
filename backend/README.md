# Backend Skeleton

FastAPI backend contracts for frontend demo integration.

## Start

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Env

- `MODEL_PROVIDER` default: `placeholder`
- `CORS_ORIGIN` default: `http://localhost:5173`

## Endpoints

- `GET /health`
- `POST /api/search`
- `POST /api/search-by-audio`
- `POST /api/classify-audio`

With `MODEL_PROVIDER=placeholder`, all three `/api/*` endpoints return:
- `501 Not Implemented`
- `{ "code": "MODEL_NOT_CONNECTED", "message": "..." }`

Later, implement a real provider and switch via env without changing endpoint contracts.
