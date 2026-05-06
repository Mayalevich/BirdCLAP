import type { ClassificationHit, SearchResult, VocabMode } from "./types";
import { hydrateResultCache, persistResultCache } from "./resultCachePersistence";
import { loadSaved } from "@/saved/savedStore";
import { ApiNotConfiguredError } from "@/lib/demoErrors";

export { formatUserFacingDemoError } from "@/lib/demoErrors";

interface BackendError {
  code?: string;
  message?: string;
}

interface SearchApiItem {
  id: string;
  title: string;
  species: string;
  score: number;
  recording_id?: string;
  scientific_name?: string;
  vocalization_type?: string;
  duration?: string;
  image_url?: string;
  audio_url?: string;
  species_code?: string;
  species_description?: string;
}

interface DescribeApiResponse {
  descriptions: string[];
}

interface SearchApiResponse {
  query: string;
  count: number;
  results: SearchApiItem[];
}

interface ClassifyApiResponse {
  count: number;
  results: ClassificationHit[];
}

/** In-memory cache hydrated from localStorage on load; search/compare/viz use this + saved list. */
const resultCache = hydrateResultCache();

/** API origin without trailing slash, or empty string = same-origin (Vite dev proxy → FastAPI). */
function getApiOriginForFetch(): string {
  if (import.meta.env.DEV) {
    return "";
  }
  const raw = import.meta.env.VITE_API_BASE_URL;
  if (!raw || !String(raw).trim()) {
    throw new ApiNotConfiguredError();
  }
  return String(raw).replace(/\/+$/, "");
}

/** Path must start with `/` (e.g. `/api/search`). Empty origin → relative URL uses Vite proxy in dev. */
function apiFetchUrl(path: string): string {
  const origin = getApiOriginForFetch();
  const p = path.startsWith("/") ? path : `/${path}`;
  return origin ? `${origin}${p}` : p;
}

/** Dev: assumed OK via `vite.config` proxy (backend on 127.0.0.1:8000). Prod/preview: needs `VITE_API_BASE_URL`. */
export function isApiConfigured(): boolean {
  return Boolean(import.meta.env.DEV) || Boolean(import.meta.env.VITE_API_BASE_URL?.trim());
}

function placeholderAvatar(name: string): string {
  const q = encodeURIComponent(name || "Unknown species");
  return `https://ui-avatars.com/api/?name=${q}&size=128&background=1a5f4a&color=fff&bold=true`;
}

async function parseErrorMessage(response: Response): Promise<string> {
  let apiError: BackendError | null = null;
  try {
    apiError = (await response.json()) as BackendError;
  } catch {
    apiError = null;
  }

  if (response.status === 501) {
    return "Demo backend is ready, but the model provider is not connected yet.";
  }
  return apiError?.message || `Request failed (${response.status}).`;
}

function mapItem(row: SearchApiItem): SearchResult {
  return {
    id: row.id,
    recordingId: row.recording_id || row.id,
    commonName: row.title || row.species || "Unknown",
    scientificName: row.scientific_name || row.species || row.title || "Unknown",
    vocalizationType: row.vocalization_type || "unknown",
    duration: row.duration || "unknown",
    speciesCode: row.species_code || "unknown",
    similarity: row.score,
    imageUrl: row.image_url || placeholderAvatar(row.title || row.species || "Unknown"),
    audioUrl: row.audio_url,
    speciesDescription: row.species_description || undefined,
  };
}

function validateMappedResult(row: SearchResult, rawId: string): SearchResult | null {
  if (!row.id || !row.recordingId || !row.commonName || !row.imageUrl) {
    console.warn("[api] Skipping search row with missing required fields:", rawId);
    return null;
  }
  return row;
}

function cacheResults(results: SearchResult[]): void {
  results.forEach((r) => resultCache.set(r.id, r));
  persistResultCache(resultCache);
}

/** Persist a single catalog row (e.g. when user adds to Compare) so refresh still resolves slots. */
export function rememberResult(row: SearchResult): void {
  if (!row?.id) return;
  resultCache.set(row.id, row);
  persistResultCache(resultCache);
}

export async function searchDataset(query: string, _vocab: VocabMode): Promise<SearchResult[]> {
  const response = await fetch(apiFetchUrl("/api/search"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 10 }),
  });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as SearchApiResponse;
  const raw = Array.isArray(data.results) ? data.results : [];
  const results: SearchResult[] = [];
  for (const item of raw) {
    const mapped = validateMappedResult(mapItem(item), item?.id ?? "?");
    if (mapped) results.push(mapped);
  }
  cacheResults(results);
  return results;
}

export async function searchSimilarToUpload(file: File): Promise<SearchResult[]> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", "10");
  const response = await fetch(apiFetchUrl("/api/search-by-audio"), { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as SearchApiResponse;
  const raw = Array.isArray(data.results) ? data.results : [];
  const results: SearchResult[] = [];
  for (const item of raw) {
    const mapped = validateMappedResult(mapItem(item), item?.id ?? "?");
    if (mapped) results.push(mapped);
  }
  cacheResults(results);
  return results;
}

function normalizeClassifyHits(rows: unknown[]): ClassificationHit[] {
  const out: ClassificationHit[] = [];
  for (const r of rows) {
    if (!r || typeof r !== "object") continue;
    const h = r as ClassificationHit;
    if (typeof h.label !== "string" || !h.label.trim()) continue;
    const score = typeof h.score === "number" && !Number.isNaN(h.score) ? h.score : 0;
    out.push({
      label: h.label.trim(),
      scientificName: typeof h.scientificName === "string" ? h.scientificName : undefined,
      score,
    });
  }
  return out;
}

export async function classifyUpload(file: File): Promise<ClassificationHit[]> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", "5");
  const response = await fetch(apiFetchUrl("/api/classify-audio"), { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as ClassifyApiResponse;
  const raw = Array.isArray(data.results) ? data.results : [];
  return normalizeClassifyHits(raw);
}

export function getResultById(id: string): SearchResult | undefined {
  const cached = resultCache.get(id);
  if (cached) return cached;
  return loadSaved().find((r) => r.id === id);
}

/**
 * POST /api/describe-audio — returns acoustic text descriptions that CLAP associates
 * with the uploaded clip. Species names are scrubbed; strings describe sound texture,
 * pitch, rhythm, and pattern. Returns [] if the endpoint is unavailable or the
 * description gallery hasn't been built.
 */
export async function describeAudio(file: File): Promise<string[]> {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("top_k", "4");
    const response = await fetch(apiFetchUrl("/api/describe-audio"), {
      method: "POST",
      body: formData,
    });
    if (!response.ok) return [];
    const data = (await response.json()) as DescribeApiResponse;
    return Array.isArray(data.descriptions) ? data.descriptions.filter((d) => d.trim()) : [];
  } catch {
    return [];
  }
}

/** GET /health — responds instantly (no model load) and is used for the status badge. */
export async function probeBackendReachable(signal?: AbortSignal): Promise<boolean> {
  if (!isApiConfigured()) return false;
  try {
    const response = await fetch(apiFetchUrl("/health"), { method: "GET", signal });
    return response.ok;
  } catch {
    return false;
  }
}
