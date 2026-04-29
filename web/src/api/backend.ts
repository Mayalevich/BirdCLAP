import type { ClassificationHit, SearchResult, VocabMode } from "./types";

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

const resultCache = new Map<string, SearchResult>();

function getApiBaseUrl(): string {
  const raw = import.meta.env.VITE_API_BASE_URL;
  if (!raw) {
    throw new Error("Missing VITE_API_BASE_URL. Add it to your .env file.");
  }
  return String(raw).replace(/\/+$/, "");
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
  };
}

function cacheResults(results: SearchResult[]): void {
  results.forEach((row) => resultCache.set(row.id, row));
}

export async function searchDataset(query: string, _vocab: VocabMode): Promise<SearchResult[]> {
  const response = await fetch(`${getApiBaseUrl()}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 10 }),
  });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as SearchApiResponse;
  const results = Array.isArray(data.results) ? data.results.map(mapItem) : [];
  cacheResults(results);
  return results;
}

export async function searchSimilarToUpload(file: File): Promise<SearchResult[]> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", "10");
  const response = await fetch(`${getApiBaseUrl()}/api/search-by-audio`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as SearchApiResponse;
  const results = Array.isArray(data.results) ? data.results.map(mapItem) : [];
  cacheResults(results);
  return results;
}

export async function classifyUpload(file: File): Promise<ClassificationHit[]> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", "5");
  const response = await fetch(`${getApiBaseUrl()}/api/classify-audio`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  const data = (await response.json()) as ClassifyApiResponse;
  return Array.isArray(data.results) ? data.results : [];
}

export function getResultById(id: string): SearchResult | undefined {
  return resultCache.get(id);
}
