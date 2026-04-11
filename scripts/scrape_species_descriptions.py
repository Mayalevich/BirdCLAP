"""
Multi-Source Species Description Scraper
=========================================
Scrapes vocal/sound descriptions for all species in xc_metadata_unified.csv.

Source priority by taxon:
  Birds   → AllAboutBirds (/sounds → /id → /overview) → ADW
  Mammals → Animal Diversity Web (ADW) "Communication and Perception" section
  Amphibians → Animal Diversity Web (ADW)
  Insects → Animal Diversity Web (ADW)
  Unknown → ADW → skip

Animal Diversity Web (animaldiversity.org) is the authoritative scientific
source for all non-bird taxa. It has a consistent "Communication and Perception"
section for every species account. URL pattern:
    https://animaldiversity.org/accounts/{Genus_species}/

Noise/annotation labels (Soundscape, Identity unknown, Noise, Speech, etc.)
are skipped entirely — they are not taxa and would corrupt CLAP training.

Naming variants (e.g. "Northern Raven" → "Common Raven", "Grey" → "Gray",
missing hyphens) are resolved via the NAME_ALIASES table before scraping.

Output: data/species_descriptions.json
    {
        "American Robin":    {"text": "...", "source": "allaboutbirds"},
        "Coyote":            {"text": "...", "source": "adw"},
        "Spring Peeper":     {"text": "...", "source": "adw"},
        "Soundscape":        {"text": "",   "source": "skip"},
    }

Usage:
    conda activate birdclap
    python scripts/scrape_species_descriptions.py
    python scripts/scrape_species_descriptions.py --resume
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_META = Path("data/xc_metadata_unified.csv")
CLASSES_CSV  = Path("data/classes.csv")
OUT_PATH     = Path("data/species_descriptions.json")

HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; BirdCLAP research scraper)"}
AAB_DELAY  = 1.2
ADW_DELAY  = 0.8

AAB_GUIDE  = "https://www.allaboutbirds.org/guide/{slug}/{tab}"
ADW_BASE   = "https://animaldiversity.org/accounts/{sci}/"

# ── labels that are not real taxa → skip from scraping and training ────────────
SKIP_LABELS = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})

# ── name aliases: metadata name → canonical AAB name ─────────────────────────
# Fixes: UK vs NA spelling, missing hyphens, subspecies → species, renamed species
NAME_ALIASES: dict[str, str] = {
    # UK Grey → NA Gray
    "Grey Catbird":               "Gray Catbird",
    "Grey Partridge":             "Gray Partridge",
    "Grey Plover":                "Black-bellied Plover",
    "Grey-cheeked Thrush":        "Gray-cheeked Thrush",
    "Grey-crowned Rosy Finch":    "Gray-crowned Rosy-Finch",
    "Grey Wagtail":               "Gray Wagtail",
    "Black-throated Grey Warbler":"Black-throated Gray Warbler",
    "Blue-grey Gnatcatcher":      "Blue-gray Gnatcatcher",
    "American Grey Flycatcher":   "Gray Flycatcher",
    "Great Grey Owl":             "Great Gray Owl",
    # "American X" → canonical AAB name
    "American Barn Owl":          "Barn Owl",
    "American Black Swift":       "Black Swift",
    "American Bushtit":           "Bushtit",
    "American Cliff Swallow":     "Cliff Swallow",
    "American Dusky Flycatcher":  "Dusky Flycatcher",
    "American Golden Plover":     "American Golden-Plover",
    "American Herring Gull":      "Herring Gull",
    "American Yellow Warbler":    "Yellow Warbler",
    # British names → North American equivalents
    "Sand Martin":                "Bank Swallow",
    "Two-barred Crossbill":       "White-winged Crossbill",
    "Common Starling":            "European Starling",
    "European Herring Gull":      "Herring Gull",
    "Common Pheasant":            "Ring-necked Pheasant",
    "Brant Goose":                "Brant",
    "Buff-bellied Pipit":         "American Pipit",
    "Black-necked Grebe":         "Eared Grebe",
    "Redwing":                    "American Robin",  # European; best available AAB fallback
    # Subspecies / splits → parent species
    "Red Fox Sparrow":            "Fox Sparrow",
    "Slate-colored Fox Sparrow":  "Fox Sparrow",
    "Sooty Fox Sparrow":          "Fox Sparrow",
    "Myrtle Warbler":             "Yellow-rumped Warbler",
    "Audubon's Warbler":          "Yellow-rumped Warbler",
    "Arctic Redpoll":             "Common Redpoll",
    # Hyphenation fixes
    "Northern Pygmy Owl":         "Northern Pygmy-Owl",
    "Western Screech Owl":        "Western Screech-Owl",
    "Eastern Screech Owl":        "Eastern Screech-Owl",
    "Northern Hawk-Owl":          "Northern Hawk Owl",
    "Eastern Wood Pewee":         "Eastern Wood-Pewee",
    "Western Wood Pewee":         "Western Wood-Pewee",
    "Pacific-slope Flycatcher":   "Pacific-Slope Flycatcher",
    "Eurasian Collared Dove":     "Eurasian Collared-Dove",
    "Leach's Storm Petrel":       "Leach's Storm-Petrel",
    # Renamed / alternate common names
    "Rock Dove":                  "Rock Pigeon",
    "Eastern Osprey":             "Osprey",
    "Western Osprey":             "Osprey",
    "Northern Raven":             "Common Raven",
    "Hudsonian Whimbrel":         "Whimbrel",
}

# Scientific name overrides for ADW lookup
ADW_SCI_OVERRIDES: dict[str, str] = {
    # Mammals
    "North American Red Squirrel": "Tamiasciurus_hudsonicus",
    "Eastern Gray Squirrel":       "Sciurus_carolinensis",
    "Eastern Chipmunk":            "Tamias_striatus",
    "Big Brown Bat":               "Eptesicus_fuscus",
    "Little Brown Myotis":         "Myotis_lucifugus",
    "Northern Hoary Bat":          "Lasiurus_cinereus",
    "Northern Myotis":             "Myotis_septentrionalis",
    "Silver-haired Bat":           "Lasionycteris_noctivagans",
    "Tricolored Bat":              "Perimyotis_subflavus",
    "Eastern Red Bat":             "Lasiurus_borealis",
    "Eastern Small-footed Myotis": "Myotis_leibii",
    "Canadian Lynx":               "Lynx_canadensis",
    "North American Porcupine":    "Erethizon_dorsatum",
    "Coyote":                      "Canis_latrans",
    "White-tailed Deer":           "Odocoileus_virginianus",
    "Collared Pika":               "Ochotona_collaris",
    "Singing Vole":                "Microtus_miurus",
    "Taiga Vole":                  "Microtus_gregalis",
    # Amphibians
    "Spring Peeper":               "Pseudacris_crucifer",
    "Wood Frog":                   "Lithobates_sylvaticus",
    "American Bullfrog":           "Lithobates_catesbeianus",
    "American Toad":               "Anaxyrus_americanus",
    "Canadian Toad":               "Anaxyrus_hemiophrys",
    "Bronze Frog":                 "Lithobates_clamitans",
    "Boreal Chorus Frog":          "Pseudacris_maculata",
    "Eastern Gray Treefrog":       "Hyla_versicolor",
    "Gray Treefrog":               "Hyla_versicolor",
    "Green Frog":                  "Lithobates_clamitans",
    "Northern Leopard Frog":       "Lithobates_pipiens",
    "Pacific Chorus Frog":         "Pseudacris_regilla",
    "Pickerel Frog":               "Lithobates_palustris",
    "Great Basin Spadefoot":       "Spea_intermontana",
    "Great Plains Toad":           "Anaxyrus_cognatus",
    "Western Toad":                "Anaxyrus_boreas",
    "Plains Spadefoot Toad":       "Spea_bombifrons",
    # Birds (scientific name disambiguation)
    "Steller's Sea Eagle":         "Haliaeetus_pelagicus",
}

# ── AllAboutBirds ─────────────────────────────────────────────────────────────

def _aab_slug(name: str) -> str:
    return name.replace(" ", "_").replace("'", "").replace("\u2019", "")


def _aab_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for p in soup.find_all("p"):
        t = p.get_text(strip=True)
        if len(t) < 60:
            continue
        tl = t.lower()
        if any(x in tl for x in ("cornell lab will send you", "get a customized list",
                                  "explore birds based on", "allaboutbirds.org")):
            continue
        if re.search(r"\(order:.*family:", tl):
            continue
        out.append(t)
    return out


def _aab_fetch_tab(slug: str, tab: str) -> str | None:
    url = AAB_GUIDE.format(slug=slug, tab=tab)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    if "news/search" in r.url or "/search?" in r.url:
        return None
    paras = _aab_paragraphs(r.text)
    return " ".join(paras) if paras else None


def scrape_allaboutbirds(common_name: str) -> str | None:
    """Try canonical name then aliases. /sounds → /id → /overview."""
    names_to_try = [common_name]
    alias = NAME_ALIASES.get(common_name)
    if alias and alias != common_name:
        names_to_try.append(alias)

    for name in names_to_try:
        slug = _aab_slug(name)
        variants = [slug]
        compact = slug.replace("-", "")
        if compact != slug:
            variants.append(compact)

        for tab in ("sounds", "id", "overview"):
            for s in variants:
                text = _aab_fetch_tab(s, tab)
                if text:
                    return text
            time.sleep(AAB_DELAY)

    return None


# ── Animal Diversity Web ──────────────────────────────────────────────────────

def _adw_sci_slug(common_name: str, scientific_name: str) -> str | None:
    """Return Genus_species slug for ADW URL, preferring known overrides."""
    if common_name in ADW_SCI_OVERRIDES:
        return ADW_SCI_OVERRIDES[common_name]
    if scientific_name and " " in scientific_name.strip():
        parts = scientific_name.strip().split()
        return f"{parts[0]}_{parts[1]}"
    return None


def scrape_adw(common_name: str, scientific_name: str = "") -> str | None:
    """Scrape Animal Diversity Web 'Communication and Perception' section."""
    slug = _adw_sci_slug(common_name, scientific_name)
    if not slug:
        return None

    url = ADW_BASE.format(sci=slug)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
    except requests.RequestException:
        return None

    if r.status_code != 200:
        return None

    # Check we didn't get redirected to a search page
    if "search" in r.url.lower() or "404" in r.url:
        return None

    time.sleep(ADW_DELAY)
    soup = BeautifulSoup(r.text, "html.parser")

    # Find "Communication and Perception" section heading
    comm_heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and "communication" in tag.get_text(strip=True).lower()
    )
    if not comm_heading:
        return None

    texts = []
    for sib in comm_heading.find_next_siblings():
        if sib.name in ("h2", "h3", "h4"):
            break
        t = sib.get_text(" ", strip=True)
        # Strip junk run-on labels like "Communication Channelsacousticacoustic"
        t = re.sub(r"(Communication Channels|Perception Channels|Other Communication Modes)"
                   r"[a-zA-Z\s]+", "", t).strip()
        if len(t) > 40:
            texts.append(t)

    return " ".join(texts) if texts else None


# ── species list ──────────────────────────────────────────────────────────────

def _load_bird_names(classes_csv: Path) -> set[str]:
    birds = set()
    if not classes_csv.exists():
        return birds
    with open(classes_csv) as f:
        for row in csv.DictReader(f):
            name = str(row.get("Name", "")).strip()
            if name:
                birds.add(name)
    return birds


def load_species(metadata_paths: list[str],
                 classes_csv: Path) -> list[tuple[str, str, bool]]:
    """
    Returns sorted list of (common_name, scientific_name, is_bird).
    is_bird = appears in classes.csv (HawkEars vocabulary).
    """
    bird_sci: dict[str, str] = {}
    if classes_csv.exists():
        with open(classes_csv) as f:
            for row in csv.DictReader(f):
                n = str(row.get("Name", "")).strip()
                s = str(row.get("AltName", "")).strip()
                if n:
                    bird_sci[n] = s

    all_species: dict[str, str] = {}
    for path in metadata_paths:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
        sci_col  = next((c for c in ("scientific_name", "sci_name") if c in df.columns), None)
        if name_col is None:
            continue
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            sci  = str(row[sci_col]).strip() if sci_col and pd.notna(row.get(sci_col)) else ""
            if name and name.lower() not in SKIP_LABELS and name not in all_species:
                all_species[name] = sci

    result = []
    seen = set()
    for name, sci in sorted(all_species.items()):
        sci_final = bird_sci.get(name, sci)
        is_bird   = name in bird_sci
        result.append((name, sci_final, is_bird))
        seen.add(name)

    # Include classes.csv birds not in any metadata
    for name, sci in sorted(bird_sci.items()):
        if name not in seen:
            result.append((name, sci, True))

    return sorted(result, key=lambda x: x[0])


# ── main ──────────────────────────────────────────────────────────────────────

# ── hand-written fallbacks for species with no scrapable source ───────────────
# Only used when all automated scraping fails. Kept minimal and factual.
MANUAL_TEXT: dict[str, str] = {
    "Leach's Storm Petrel": (
        "Leach's Storm-Petrel produces a variety of eerie, chattering calls at "
        "breeding colonies, including a sustained rhythmic chatter given in flight "
        "and a purring call from within the burrow. They are generally silent at sea."
    ),
    "Marsh ground cricket": (
        "The marsh ground cricket produces a continuous, high-pitched trilling song, "
        "typically a sustained buzz or trill delivered from low vegetation near water."
    ),
    "Riley's tree cricket": (
        "Riley's tree cricket produces a sustained, melodious trilling song, "
        "a clear high-pitched pulse repeated at a regular rate, characteristic "
        "of tree crickets in the genus Oecanthus."
    ),
    "Steindachner's Shieldback": (
        "Steindachner's Shieldback katydid produces broadband stridulatory signals "
        "with peak frequencies of 11–17 kHz, generated by alternating pulse trains "
        "during tegmina opening and closing movements."
    ),
    "Singing Vole": (
        "Singing Voles are unusually vocal for small rodents, producing sequences "
        "of high-pitched calls and chatters used for territorial advertisement and "
        "social communication, audible from a distance."
    ),
    "Taiga Vole": (
        "Taiga Voles produce high-pitched squeaks and chatters for social "
        "communication, similar to other Microtus voles, though detailed "
        "acoustic studies are limited."
    ),
    "Yellow-browed Warbler": (
        "Yellow-browed Warbler gives a loud, rising 'tseeweest' call and a "
        "high-pitched medley of whistles in song, often described as penetrating "
        "and distinctly two-syllabled."
    ),
    "Pink-footed Shearwater": (
        "Pink-footed Shearwater gives horse-like whinny calls and raucous "
        "cackling at breeding colonies. At sea they are largely silent. "
        "Duet calls between mates are given from within nesting burrows."
    ),
    "Tricolored Bat": (
        "Tricolored Bat (formerly Eastern Pipistrelle) uses ultrasonic echolocation "
        "calls around 45 kHz for navigation and hunting. Social calls include "
        "soft chips and chirps. It is one of the smallest bats in North America."
    ),
    "Western Toad": (
        "Western Toad produces a soft, bird-like peeping or chirping call rather "
        "than the loud advertisement calls typical of many frogs. Males call from "
        "shallow water during the spring breeding season."
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+", default=[str(UNIFIED_META)])
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--resume", action="store_true",
                        help="Skip species already in output with non-empty text")
    parser.add_argument("--adw-only", action="store_true",
                        help="Skip AllAboutBirds, use ADW for all species")
    parser.add_argument("--rescrape-source", metavar="SOURCE",
                        help="Re-scrape only entries currently from this source "
                             "(e.g. 'wikipedia', 'failed'). Ignores --resume for "
                             "matching entries.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if (args.resume or args.rescrape_source) and out_path.exists():
        existing = json.loads(out_path.read_text())
        done = sum(1 for v in existing.values() if v.get("text", "").strip())
        print(f"Loaded existing — {done}/{len(existing)} entries with text")
        if args.rescrape_source:
            targets = {k for k, v in existing.items()
                       if v.get("source") == args.rescrape_source}
            print(f"Will re-scrape {len(targets)} entries with source='{args.rescrape_source}'")

    species_list = load_species(args.metadata, CLASSES_CSV)
    print(f"Total species to process: {len(species_list)}")
    birds    = sum(1 for _, _, b in species_list if b)
    nonbirds = len(species_list) - birds
    print(f"  Birds (AllAboutBirds → ADW): {birds}")
    print(f"  Non-birds (ADW): {nonbirds}\n")

    results = dict(existing)
    counts  = {"allaboutbirds": 0, "adw": 0, "manual": 0, "failed": 0}

    for i, (name, sci, is_bird) in enumerate(species_list, 1):
        # Hard skip for noise/annotation labels
        if name.lower() in SKIP_LABELS:
            results[name] = {"text": "", "source": "skip"}
            continue

        # Determine whether to skip this entry
        already_done = name in results and results[name].get("text", "").strip()
        is_rescrape_target = args.rescrape_source and results.get(name, {}).get("source") == args.rescrape_source

        if args.resume and already_done and not is_rescrape_target:
            continue
        if not args.resume and not args.rescrape_source:
            pass  # process everything
        elif args.rescrape_source and not is_rescrape_target and already_done:
            continue  # skip entries that are NOT the target source

        text   = None
        source = ""

        # ── Manual fallback text (for species with no scrapable source) ──
        if name in MANUAL_TEXT:
            text   = MANUAL_TEXT[name]
            source = "manual"

        # ── AllAboutBirds ──
        if text is None and not args.adw_only:
            text = scrape_allaboutbirds(name)
            if text:
                source = "allaboutbirds"

        # ── ADW fallback ──
        if text is None:
            text = scrape_adw(name, sci)
            if text:
                source = "adw"

        if text:
            results[name] = {"text": text, "source": source}
            counts[source] = counts.get(source, 0) + 1
            flag = " [UPGRADED]" if is_rescrape_target else ""
            print(f"  [{i:4d}/{len(species_list)}] {name}: {len(text)} chars [{source.upper()}]{flag}")
        else:
            # Keep existing text if re-scrape found nothing better
            if is_rescrape_target and already_done:
                print(f"  [{i:4d}/{len(species_list)}] {name}: no upgrade found, keeping existing")
            else:
                results[name] = {"text": "", "source": "failed"}
                counts["failed"] += 1
                print(f"  [{i:4d}/{len(species_list)}] {name}: NO TEXT FOUND")

        if i % 10 == 0:
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'─'*60}")
    print(f"Done. {len(results)} species → {out_path}")
    # Full tally across all entries (not just this run)
    all_sources = {}
    for v in results.values():
        s = v.get("source", "failed")
        all_sources[s] = all_sources.get(s, 0) + 1
    for src, cnt in sorted(all_sources.items(), key=lambda x: -x[1]):
        print(f"  {src:<16}: {cnt}")
    failed_list = [n for n, v in results.items() if v.get("source") == "failed"]
    if failed_list:
        print(f"\n  Still no text for:")
        for n in sorted(failed_list):
            print(f"    {n}")


if __name__ == "__main__":
    main()
