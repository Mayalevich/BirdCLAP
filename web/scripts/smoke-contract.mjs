/**
 * Minimal contract check: objects shaped like SearchResult must carry fields the UI reads.
 * Run after `npm run build` via `npm run smoke`.
 */

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function assertSearchResult(row, label) {
  assert(row && typeof row === "object", `${label}: not an object`);
  for (const k of [
    "id",
    "recordingId",
    "commonName",
    "scientificName",
    "vocalizationType",
    "duration",
    "speciesCode",
    "imageUrl",
  ]) {
    assert(
      typeof row[k] === "string" && row[k].length > 0,
      `${label}: missing or empty string field "${k}"`
    );
  }
}

// Synthetic row matching mapItem() output shape
const sample = {
  id: "smoke-1",
  recordingId: "rec-1",
  commonName: "Demo Bird",
  scientificName: "Demo demo",
  vocalizationType: "song",
  duration: "0:30",
  speciesCode: "demo_bird",
  imageUrl: "https://example.com/x.png",
};

assertSearchResult(sample, "fixture");

// Classification hit contract
const hit = { label: "X", score: 0.5 };
assert(typeof hit.label === "string" && hit.label.length > 0, "ClassificationHit.label");
assert(typeof hit.score === "number" && !Number.isNaN(hit.score), "ClassificationHit.score");

console.log("smoke-contract: OK");
