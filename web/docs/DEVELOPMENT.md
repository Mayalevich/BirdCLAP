# Web — development guide

## Prerequisites

- **Node.js** 18+ (LTS recommended)
- **npm** (comes with Node)

## Commands

From the `web/` directory:

```bash
npm install          # once per clone / after dependency changes
npm run dev          # Vite dev server (HMR)
npm run build        # TypeScript check + production bundle
npm run smoke        # build + lightweight API shape contract script
npm run preview      # Serve the production build locally
```

There is no separate `eslint` script in `package.json` today; **`npm run build`** runs **`tsc --noEmit`** and will catch type errors.

## Environment variables

During **`npm run dev`**, the SPA talks to **`/api/*` on the Vite origin** and **`vite.config.ts` proxies** to **`http://127.0.0.1:8000`**. You do **not** need `VITE_API_BASE_URL` in dev unless you intentionally bypass the proxy. Restart Vite after changing env or **`VITE_DEV_API_PROXY`**.

For **`vite preview`** or any static/production build: set **`VITE_API_BASE_URL`** to your public API (**no trailing slash**). Backend **CORS** must allow your site origin unless you terminate TLS on same host.

1. Reference copy: **[`.env.example`](../.env.example)**.
2. Restart **`npm run dev`** after any `.env` change affecting Vite env.

The shell shows a banner when the URL is unset, and pings **`POST /api/search`** once to populate the compact **API** badge (**API OK / API down / No API**).

## Path alias

- **`@/`** → `web/src/` (configured in `vite.config.ts`)

Imports look like: `import { searchDataset } from "@/api/backend"`.

## Adding a new route

1. Add a `<Route>` under the existing `<Route element={<AppShell />}>` in `src/App.tsx`.
2. Create a page component under `src/pages/`.
3. Add a `<NavLink>` in `src/layout/AppShell.tsx` if it should appear in the main nav.

## Dependencies worth knowing

| Package | Use |
|---------|-----|
| `react`, `react-dom` | UI |
| `react-router-dom` | SPA routing |
| `three` | WebGL + postprocessing (viz chunk) |

## Production build notes

- **`BirdSoundEmbeddingViz`** is **lazy-loaded** from `VizPage.tsx` to keep the initial bundle smaller; editing it affects a separate chunk.
- Build may warn about chunk size for the viz bundle; that is expected while Three + postprocessing are bundled together.

## Styling

- Global rules and page-specific classes live in **`src/index.css`** (no CSS-in-JS, no Tailwind).
- Prefer reusing existing utility patterns: `.panel`, `.page-header`, `.muted`, `.btn`, etc.

## API client and mocks

| Module | Purpose |
|--------|---------|
| **`src/api/backend.ts`** | Live development client used by **`QueryPage`**, **`ComparePage`**, **`VizPage`**. Persists fetched rows to `localStorage` so ids resolve after reload. |
| **`src/api/mock.ts`** | Deterministic fixtures for offline work or prototyping. |

Keeping **`SearchResult`** in **`src/api/types.ts`** stable makes backend swaps easier — add adapters rather than rewriting cards.
