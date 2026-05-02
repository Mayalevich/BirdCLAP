"""
CLAP Text Description Generator (RAG + LLM API)
================================================
For each unique (species, vocalization_type) combination in the audio dataset,
generates 4 varied natural language descriptions grounded in species expert text.
These become the text side of CLAP (audio, text) training pairs.

RAG approach: species description is fed as context so the model paraphrases
a verified source rather than generating from memory.

Supports **OpenAI** (default) or **Google Gemini** via `--provider gemini`.

Descriptions are per (species, type) — all clips of the same combo share the
same 4 descriptions. The multi-positive contrastive loss in train_clap.py
handles this correctly by treating all same-combo clips as joint positives,
so there is no false negative problem.

Inputs:
    data/species_descriptions.json  — expert text per species (multi-source)
    data/xc_metadata_unified.csv    — (or any metadata CSV with common_name + vocalization_type)

Output:
    data/clap_descriptions.json  (written after each successful species|type; use --output to change)
    {
        "American Robin||song": [
            "American Robin singing its rich, melodic series of flute-like phrases...",
            ...
        ],
        ...
    }

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY=sk-...
    python scripts/generate_clap_descriptions.py

    # Google Gemini — same prompts and output format as OpenAI (default --delay is 5 s; use --delay 12–20 on free tier if needed)
    export GEMINI_API_KEY=...   # or GOOGLE_API_KEY (Google AI Studio)
    pip install google-generativeai
    python scripts/generate_clap_descriptions.py --provider gemini --gemini-model gemini-2.0-flash

    # Only combos that still lack rich label variants (typically ≤5 taxonomy-style lines).
    python scripts/generate_clap_descriptions.py --provider gemini --taxonomy-gap-only \\
        --labels data/clap_all_labels.json

    Re-running skips (species, type) keys that already have 4 descriptions
    in clap_descriptions.json. Use --no-skip-existing to regenerate everything.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from openai import APIStatusError

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", encoding="utf-8-sig")
except ImportError:
    pass

AAB_PATH   = Path("data/species_descriptions.json")
TAX_PATH   = Path("data/species_taxonomy.json")
OUT_PATH   = Path("data/clap_descriptions.json")
MODEL      = "gpt-4o-mini"   # cheap + fast; swap to "gpt-4o" for higher quality
# Default is a light throttle; 429 retries back off automatically. Use e.g. --delay 21
# if your org is still capped at ~3 RPM on gpt-4o-mini.
DEFAULT_DELAY = 0.35
# Gemini free tier RPM is often very low; spacing between *successful* requests matters.
# Use --delay to override; see also --delay default when --provider gemini.
DEFAULT_DELAY_GEMINI = 5.0

# Vocalization types that are noisy/uninformative — skip these
SKIP_TYPES = {"uncertain", "various", "various calls", "nan", ""}

SYSTEM_PROMPT = """You are writing acoustic training text for a contrastive audio-text model.
Your output must NEVER name the species, genus, Latin names, or any taxonomic identifier.
Refer only with "this species", "the bird", or "the animal."

Focus ONLY on observable sound attributes grounded in the source text:
• sound texture (whistled, buzzy, raspy, clear, nasal),
• rhythm and repetition (rapid phrases, spaced notes, accelerating trills),
• pitch contours (rising, falling, high, low, modulated),
• timing (short, sustained),
• behavioural context WHEN the source mentions it — do not invent.

Rules:
• Base EVERY acoustic detail on the supplied species/source text — no invented ecology.
• If the source does NOT cover this vocalization type, reply with exactly INSUFFICIENT_SOURCE_DATA.
• Exactly 4 outputs: each 1–2 sentences, ~15–40 words.
• Avoid repetitive openings; do not lecture the listener.
• Do NOT mention microphones, bitrate, habitats, geography, seasons, recording quality."""

USER_TEMPLATE = """Vocalization type (context only — do NOT echo the species label in prose): {voc_type}

Expert source excerpt:
\"\"\"
{aab_text}
\"\"\"

Produce exactly four acoustic descriptions referring only to pitch, rhythm,
timbre, and pattern. Reply with ONLY a JSON array of 4 strings, no preamble.
Example format: ["desc1", "desc2", "desc3", "desc4"]"""


def scrub_species_name(text: str, common: str, scientific: str) -> str:
    """Rewrite accidental species leakage to generic references (cheap safety net)."""
    out = text
    common = common.strip()
    scientific = scientific.strip()
    if common:
        out = re.sub(re.escape(common), "this species", out, flags=re.IGNORECASE)
    if scientific:
        out = re.sub(re.escape(scientific), "this species", out, flags=re.IGNORECASE)
        for tok in scientific.split():
            if len(tok) > 3:
                out = re.sub(
                    rf"\b{re.escape(tok)}\b", "this species", out,
                    flags=re.IGNORECASE,
                )
    return out


def _entry_complete(value: object) -> bool:
    """True if this output key already has a full set of generated strings."""
    if not isinstance(value, list) or len(value) < 4:
        return False
    return all(isinstance(s, str) and s.strip() for s in value)


def _parse_four_descriptions(
    raw: str | None,
    common_name: str,
    voc_type: str,
) -> list[str] | None:
    """Extract a JSON array of strings from LLM output. Returns None on skip/parse failure."""
    if raw is None or not isinstance(raw, str):
        print(f"  [parse error] {common_name} | {voc_type}: empty response")
        return None
    text = raw.strip()

    if "INSUFFICIENT_SOURCE_DATA" in text:
        return None

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  [parse error] {common_name} | {voc_type}: {text[:80]}")
        return None
    try:
        descs = json.loads(text[start:end])
    except json.JSONDecodeError:
        print(f"  [parse error] JSON: {common_name} | {voc_type}")
        return None
    if isinstance(descs, list) and all(isinstance(d, str) for d in descs) and len(descs) >= 4:
        return descs
    print(f"  [parse error] expected ≥4 strings: {common_name} | {voc_type}")
    return None


def get_combos(metadata_paths: list[str]) -> list[tuple[str, str]]:
    """Extract unique (common_name, type) pairs from one or more metadata CSVs."""
    combos = set()
    for path in metadata_paths:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        if "common_name" not in df.columns:
            print(f"  Skipping {path} — missing common_name column")
            continue
        if "type" not in df.columns and "vocalization_type" in df.columns:
            df = df.rename(columns={"vocalization_type": "type"})
        if "type" not in df.columns:
            print(f"  Skipping {path} — missing type or vocalization_type column")
            continue
        for _, row in df.iterrows():
            name = str(row["common_name"]).strip()
            for t in str(row["type"]).split(","):
                t = t.strip().lower()
                if t not in SKIP_TYPES and name and name != "nan":
                    combos.add((name, t))
    return sorted(combos)


def _wait_seconds_from_429(message: str) -> float:
    m = re.search(r"try again in (\d+)s", str(message), re.I)
    if m:
        return float(m.group(1)) + 2.0
    return 25.0


def _openai_error_code(e: APIStatusError) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("code"):
            return str(err["code"])
        if body.get("code"):
            return str(body["code"])
    if getattr(e, "code", None):
        return str(e.code)
    return ""


def generate_descriptions(
    client: OpenAI,
    common_name: str,
    voc_type: str,
    aab_text: str,
    scientific_name: str = "",
    max_retries: int = 12,
) -> list[str] | None:
    """Call OpenAI to generate 4 descriptions. Retries on RPM rate limits."""
    prompt = USER_TEMPLATE.format(
        voc_type=voc_type,
        aab_text=aab_text[:2000],
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            content = getattr(resp.choices[0].message, "content", None)
            text = (content or "").strip()
            descs = _parse_four_descriptions(text, common_name, voc_type)
            if not descs:
                return None
            return [
                scrub_species_name(d, common_name, scientific_name)
                for d in descs
            ]
        except APIStatusError as e:
            code = _openai_error_code(e)
            if e.status_code == 429 and code == "insufficient_quota":
                print(f"  [API error] {common_name} | {voc_type}: {e}")
                print("\nOpenAI quota/billing — add credits, then re-run (progress is auto-saved).")
                sys.exit(1)
            if e.status_code == 429:
                wait = _wait_seconds_from_429(str(e))
                print(f"  [429 rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})…")
                time.sleep(wait)
                continue
            print(f"  [API error] {common_name} | {voc_type}: {e}")
            return None
        except Exception as e:
            print(f"  [API error] {common_name} | {voc_type}: {e}")
            return None
    print(f"  [give up] {common_name} | {voc_type}: too many 429 retries")
    return None


def _gemini_response_text(resp: object) -> str:
    """Extract plain text from a google-generativeai response (handles BLOCKED finishes)."""
    try:
        t = getattr(resp, "text", None)
        if t:
            return str(t).strip()
    except Exception:
        pass
    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        parts = getattr(cands[0].content, "parts", None)
        if not parts:
            return ""
        bits: list[str] = []
        for p in parts:
            bits.append(str(getattr(p, "text", "") or ""))
        return "".join(bits).strip()
    except Exception:
        return ""

def _gemini_wait_seconds(attempt: int, exc: BaseException | None) -> float:
    """
    Seconds to wait before retrying a Gemini request after rate limit / quota errors.
    Parses server hints when possible; otherwise exponential backoff + jitter.
    """
    if exc is not None:
        msg = str(exc)
        m = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", msg, re.I)
        if m:
            return float(m.group(1)) + 1.5 + random.uniform(0, 1.0)
        m = re.search(r"(\d+(?:\.\d+)?)\s*s\s*timeout", msg, re.I)
        if m:
            return float(m.group(1)) + 2.0 + random.uniform(0, 2.0)
        m = re.search(r"Retry after (\d+)", msg, re.I)
        if m:
            return float(m.group(1)) + 2.0 + random.uniform(0, 1.0)
    # Fallback: cap at 3 min; Gemini free tier benefits from long pauses
    base = min(180.0, 20.0 * (1.55**attempt))
    return base + random.uniform(0, min(8.0, base * 0.15))


def _is_gemini_transient_quota(e: BaseException) -> bool:
    """429 / resource exhausted / RPM — worth retrying."""
    name = type(e).__name__
    if name in ("ResourceExhausted", "TooManyRequests", "DeadlineExceeded"):
        return True
    err = str(e).lower()
    checks = (
        "429",
        "resource",
        "quota",
        "rate",
        "ratelimit",
        "capacity",
        "exhausted",
        "throttl",
        "try again",
        "too many requests",
        "overload",
        "busy",
        "suspend",
        "upstream",
        "retry",
        "temporar",
    )
    return any(x in err for x in checks)


def generate_descriptions_gemini(
    genai_pkg: Any,
    model_obj: Any,
    common_name: str,
    voc_type: str,
    aab_text: str,
    scientific_name: str,
    temperature: float,
    max_retries: int = 24,
) -> list[str] | None:
    """Call Google Gemini to generate 4 descriptions. Retries on rate limits."""
    user_block = USER_TEMPLATE.format(
        voc_type=voc_type,
        aab_text=aab_text[:2000],
    )
    prompt = SYSTEM_PROMPT.strip() + "\n\n" + user_block
    gc = genai_pkg.types.GenerationConfig(
        max_output_tokens=1024,
        temperature=max(0.0, min(2.0, temperature)),
    )

    for attempt in range(max_retries):
        try:
            response = model_obj.generate_content(
                prompt,
                generation_config=gc,
            )
            text = _gemini_response_text(response)
            if not text.strip():
                fb = getattr(response, "prompt_feedback", None)
                print(f"  [Gemini empty/finish reason] {common_name} | {voc_type}: {fb}")
                return None
            descs = _parse_four_descriptions(text, common_name, voc_type)
            if not descs:
                return None
            return [
                scrub_species_name(d, common_name, scientific_name)
                for d in descs
            ]
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if _is_gemini_transient_quota(e):
                wait = _gemini_wait_seconds(attempt, e)
                print(
                    f"  [Gemini backoff] {common_name} | {voc_type}: "
                    f"wait {wait:.1f}s (try {attempt + 1}/{max_retries}) "
                    f"[{type(e).__name__}]"
                )
                time.sleep(wait)
                continue
            print(f"  [API error] Gemini {common_name} | {voc_type}: {e}")
            return None
    print(f"  [give up] Gemini {common_name} | {voc_type}: too many retries")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+",
                        default=["data/xc_metadata_unified.csv"],
                        help="One or more metadata CSVs with common_name and vocalization_type columns")
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--aab", default=str(AAB_PATH),
                        help="Species descriptions JSON")
    parser.add_argument(
        "--taxonomy",
        default=str(TAX_PATH),
        help="species_taxonomy.json for scientific-name scrub pass (default data/species_taxonomy.json)",
    )
    parser.add_argument(
        "--provider", choices=("openai", "gemini"), default="openai",
        help="LLM backend: OpenAI Chat Completions or Google Gemini API (default: openai)",
    )
    parser.add_argument(
        "--gemini-model", default="gemini-2.0-flash",
        help="Gemini model id for --provider gemini (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--gemini-temperature", type=float, default=0.45,
        help="Gemini sampling temperature (default 0.45; lower = more conservative).",
    )
    parser.add_argument(
        "--gemini-max-retries", type=int, default=24,
        help="Max retries per combo when Gemini returns rate/quota errors (default 24).",
    )
    parser.add_argument(
        "--taxonomy-gap-only", action="store_true",
        help="Only process combos that have ≤ --taxonomy-gap-max-variants label lines "
             "in --labels (focuses generation on taxonomy-rich pools missing GPT lines). "
             "Typical cutoff is 5 (five AnimalCLAP templates, no extras).",
    )
    parser.add_argument(
        "--labels", default="data/clap_all_labels.json",
        help="Label pool JSON for --taxonomy-gap-only (default: data/clap_all_labels.json)",
    )
    parser.add_argument(
        "--taxonomy-gap-max-variants", type=int, default=5,
        metavar="N",
        help="With --taxonomy-gap-only: skip combos with more than N label variants "
             "(default: 5)",
    )
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Regenerate all keys even if already present in output file")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # no-op; resume is default
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Seconds to sleep after each combo (success or skip). "
            "Default: OpenAI 0.35 s; Gemini 5.0 s (free-tier RPM is low — increase e.g. 10–20 if needed). "
            "Use 0 to disable (not recommended for Gemini)."
        ),
    )
    args = parser.parse_args()
    if args.delay is None:
        delay = DEFAULT_DELAY_GEMINI if args.provider == "gemini" else DEFAULT_DELAY
    else:
        delay = max(0.0, args.delay)

    openai_client: OpenAI | None = None
    genai_pkg: Any = None
    gem_model: Any = None

    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("ERROR: set OPENAI_API_KEY environment variable")
        openai_client = OpenAI(api_key=api_key)
    elif args.provider == "gemini":
        try:
            import google.generativeai as genai_mod
        except ImportError:
            raise SystemExit(
                "ERROR: google-generativeai not installed. Run: pip install google-generativeai"
            )
        genai_pkg = genai_mod
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key.strip():
            raise SystemExit(
                "ERROR: GEMINI_API_KEY / GOOGLE_API_KEY is unset or empty.\n"
                "  Save your `.env` if the key is only open in the editor (not written to disk), "
                "or use: `$env:GEMINI_API_KEY='…'` before running.\n"
                "  Requires one of GEMINI_API_KEY or GOOGLE_API_KEY for --provider gemini."
            )

        genai_pkg.configure(api_key=api_key.strip())
        gem_model = genai_pkg.GenerativeModel(args.gemini_model)

    tax_db: dict[str, Any] = {}
    tax_path_arg = Path(args.taxonomy)
    if tax_path_arg.is_file():
        tax_db = json.loads(tax_path_arg.read_text(encoding="utf-8"))
        print(f"Loaded taxonomy ({len(tax_db)} species) for name scrub from {tax_path_arg}")
    else:
        print(f"[warn] No taxonomy at {tax_path_arg} — scientific-name scrub is common-name only")

    raw      = json.loads(Path(args.aab).read_text(encoding="utf-8"))
    # Support both old format {"species": "text"} and new {"species": {"text": ..., "source": ...}}
    aab = {k: (v["text"] if isinstance(v, dict) else v) for k, v in raw.items()}
    out_path = Path(args.output)

    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    combos = get_combos(args.metadata)
    if args.taxonomy_gap_only:
        labels_path = Path(args.labels)
        if not labels_path.is_file():
            raise SystemExit(f"--taxonomy-gap-only requires a labels file at {labels_path}")
        lm_raw = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(lm_raw, dict):
            raise SystemExit("--labels JSON must be a dict of combo -> string list")

        thresh = args.taxonomy_gap_max_variants
        before_ct = len(combos)
        filtered: list[tuple[str, str]] = []
        for name, vtype in combos:
            key = f"{name}||{vtype}"
            lst = lm_raw.get(key)
            n = len(lst) if isinstance(lst, list) else 0
            if n <= thresh:
                filtered.append((name, vtype))
        combos = filtered
        print(
            f"--taxonomy-gap-only (≤{thresh} variants in {labels_path.name}): "
            f"{len(combos)} combos to generate (filtered from {before_ct} metadata combos)"
        )

    results = dict(existing)
    already_done = 0
    if not args.no_skip_existing:
        for name, vtype in combos:
            if _entry_complete(results.get(f"{name}||{vtype}")):
                already_done += 1

    print(f"Provider:     {args.provider}")
    print(f"Found {len(combos)} (species, type) combos queued after filters")
    if existing:
        print(f"Loaded {len(existing)} keys from {out_path}")
    if already_done and not args.no_skip_existing:
        print(f"Auto-skip {already_done} combos already complete (4 descriptions each)")
    if args.no_skip_existing:
        print("--no-skip-existing: will regenerate every combo")
    print(f"Post-request delay: {delay}s (429s/rate limits are retried)\n")

    skipped = 0

    for i, (name, vtype) in enumerate(combos, 1):
        key = f"{name}||{vtype}"
        if not args.no_skip_existing and _entry_complete(results.get(key)):
            continue

        aab_text = aab.get(name, "")
        if not aab_text:
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: no AAB text — skipping")
            skipped += 1
            continue

        sci_hint = ""
        trow = tax_db.get(name)
        if isinstance(trow, dict):
            sci_hint = str(trow.get("scientific") or "")

        if args.provider == "openai":
            assert openai_client is not None
            descs = generate_descriptions(
                openai_client, name, vtype, aab_text, scientific_name=sci_hint,
            )
        else:
            assert genai_pkg is not None and gem_model is not None
            descs = generate_descriptions_gemini(
                genai_pkg,
                gem_model,
                name,
                vtype,
                aab_text,
                scientific_name=sci_hint,
                temperature=args.gemini_temperature,
                max_retries=args.gemini_max_retries,
            )

        if descs:
            results[key] = descs
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: {len(descs)} descriptions")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(f"  [{i:3d}/{len(combos)}] {name} | {vtype}: insufficient source — skipping")
            skipped += 1

        if delay > 0:
            time.sleep(delay)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone.")
    print(f"  Saved:   {len(results)} (species, type) pairs → {out_path}")
    print(f"  Skipped: {skipped} (no AAB text or insufficient source)")
    print("\nNext step:")
    print("  python scripts/build_clap_labels.py")
    print("  python scripts/build_clap_training_pairs.py")


if __name__ == "__main__":
    raise SystemExit(main())
