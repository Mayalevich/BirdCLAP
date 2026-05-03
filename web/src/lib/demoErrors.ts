const NOT_CONFIGURED_HINT =
  "Add VITE_API_BASE_URL to web/.env (for example http://localhost:8000), then restart the dev server.";
const NETWORK_HINT =
  "Check that the API server is running and that CORS allows this site’s origin.";

export class ApiNotConfiguredError extends Error {
  override readonly name = "ApiNotConfiguredError";
  constructor() {
    super(NOT_CONFIGURED_HINT);
    Object.setPrototypeOf(this, ApiNotConfiguredError.prototype);
  }
}

export function formatUserFacingDemoError(err: unknown): string {
  if (err instanceof ApiNotConfiguredError) return `Backend URL is not set. ${NOT_CONFIGURED_HINT}`;
  if (typeof err !== "object" || err === null) return "Something went wrong. Try again.";
  const e = err as Error;
  if (e.name === "ApiNotConfiguredError") return `Backend URL is not set. ${NOT_CONFIGURED_HINT}`;

  const msg = typeof e.message === "string" ? e.message : String(err);

  if (/missing\s+VITE_API_BASE_URL|VITE_API_BASE_URL/i.test(msg)) {
    return `Backend URL is not set. ${NOT_CONFIGURED_HINT}`;
  }

  const lower = msg.toLowerCase();
  if (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("network error") ||
    lower.includes("load failed")
  ) {
    return `Could not reach the API. ${NETWORK_HINT}`;
  }

  if (/request failed \(\d+\)/i.test(msg)) {
    return `${msg} If this persists, ${NETWORK_HINT.toLowerCase()}`;
  }

  return msg.length > 180 ? `${msg.slice(0, 177)}…` : msg;
}
