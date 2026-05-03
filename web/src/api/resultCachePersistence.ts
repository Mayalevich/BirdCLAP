import type { SearchResult } from "./types";

const STORAGE_KEY = "lets-solve-it:result-cache";

export function hydrateResultCache(): Map<string, SearchResult> {
  const map = new Map<string, SearchResult>();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return map;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return map;
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (!v || typeof v !== "object") continue;
      const row = v as SearchResult;
      if (typeof row.id !== "string" || !row.id) continue;
      map.set(k, row);
    }
  } catch {
    /* ignore */
  }
  return map;
}

export function persistResultCache(map: Map<string, SearchResult>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(map)));
  } catch {
    /* quota / privacy mode */
  }
}
