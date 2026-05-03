# Web application — documentation

This folder is the **onboarding entry point** for anyone working on the `web/` Vite + React client. The Python/ML pipeline and Xeno-canto tooling live at the **repository root**; see [`CODEBASE_GUIDE.md`](../../CODEBASE_GUIDE.md) there.

## Read order for a new web developer

1. **[FEATURES.md](./FEATURES.md)** — What the app does today: routes, uploads, **`src/api/backend.ts`**, and the 3D viz.
2. **[DEVELOPMENT.md](./DEVELOPMENT.md)** — Commands, `.env`, build, aliases.
3. **[DEMO_RUNBOOK.md](./DEMO_RUNBOOK.md)** — Repeatable demo setup and quick fixes live.
4. **[DEMO_PRODUCTION_CHECKLIST.md](./DEMO_PRODUCTION_CHECKLIST.md)** — Demo-readiness backlog and team splits.

## Scope (important)

- **Backend**: search, classify, and similarity calls go to **`VITE_API_BASE_URL`** (`POST /api/search`, `/api/search-by-audio`, `/api/classify-audio`). **[`.env.example`](../.env.example)** describes the env var — copy it to **`web/.env`** and restart Vite.
- **`src/api/mock.ts`** remains useful as offline reference/fixtures only; **`QueryPage`** and related flows use **`backend.ts`** in normal development.
- **Persistence**: `localStorage` for saved specimens, vocabulary preference, and a **persisted catalog result cache** keyed by recording id (`lets-solve-it:result-cache`) so Compare/Viz lookups survive reload after a successful search.

## Source layout (quick map)

| Path | Role |
|------|------|
| `src/App.tsx` | Router + providers |
| `src/layout/AppShell.tsx` | Header, nav, API banner/badge, footer |
| `src/pages/*` | Route-level screens |
| `src/components/` | `ResultCard`, lazy `BirdSoundEmbeddingViz` |
| `src/api/backend.ts` | Live `fetch` client + `/api/search` probe + result cache hydration |
| `src/api/mock.ts` | Offline mock catalog reference |
| `src/context/` | Preferences + saved list |
| `src/saved/` | `localStorage` read/write for saved rows |
| `src/lib/` | Spectrogram + audio-driven point cloud math |
| `src/hooks/` | `useSpectrogram` |
| `src/index.css` | Global + page styles |

Questions about **training data, CLAP scripts, or CSV schema** belong in the root guide and `docs/` under the repo root, not here.
