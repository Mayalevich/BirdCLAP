# Backend Skeleton

FastAPI backend contracts for frontend demo integration.

## Start (default port **8000**)

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

**Fine-tuned CLAP (repo root, Windows):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_backend_clap.ps1
```

Defaults: `checkpoints\best.pt`, port **8000**, **`127.0.0.1`**. Before starting, the script **terminates any process LISTENing on that port** (fixes `[winerror 10048]`). To skip kills: `-DontKillListeners`.

Other flags: `-CheckpointPath`, `-Port`, `-ListenHost 0.0.0.0` (LAN).

The app **binds the HTTP port first** and loads CLAP on the first API call, so a busy port fails immediately instead of after a long GPU load.

### Port 8000 already in use (Windows)

Close old `uvicorn` terminals, then find the PID:

```powershell
netstat -ano | findstr :8000
```

End the owning process(es), e.g. `taskkill /PID <pid> /F`, then start again.

## Env

- `MODEL_PROVIDER` default: `placeholder`
- **`CORS_ORIGINS`** (comma-separated): optional explicit allow-list for the SPA.
- **`CORS_ORIGIN`** (singular): if set to `http://localhost:5173`, the backend also adds `http://127.0.0.1:5173` automatically (different browser origins).
- If neither is set, dev defaults cover **5173** and **5174** on both **`localhost`** and **`127.0.0.1`**.

## Endpoints

- `GET /health`
- `POST /api/search`
- `POST /api/search-by-audio`
- `POST /api/classify-audio`

With `MODEL_PROVIDER=placeholder`, all three `/api/*` endpoints return:
- `501 Not Implemented`
- `{ "code": "MODEL_NOT_CONNECTED", "message": "..." }`

Later, implement a real provider and switch via env without changing endpoint contracts.
