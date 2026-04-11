"""
Taxonomy Database Builder
==========================
Builds data/species_taxonomy.json — a complete Kingdom→Species lineage for
every species in the dataset, used by build_clap_labels.py to generate the
5 AnimalCLAP-style taxonomy text templates.

Scientific name resolution priority:
  1. classes.csv (HawkEars birds — most reliable)
  2. ADW_SCI_MAP (non-bird taxa with known scientific names)
  3. NAME_ALIASES → resolved canonical name → look up its scientific name
  4. GBIF species search API (last resort for anything still unmapped)

Taxonomy lookup: GBIF species match API, cached to avoid repeat requests.
For non-birds, Kingdom/Phylum/Class reflect actual taxonomy (Mammalia,
Amphibia, Insecta, etc.) rather than hardcoding Aves for everything.

Output: data/species_taxonomy.json
    {
        "American Robin": {
            "scientific": "Turdus migratorius",
            "kingdom":    "Animalia",
            "phylum":     "Chordata",
            "class":      "Aves",
            "order":      "Passeriformes",
            "family":     "Turdidae",
            "genus":      "Turdus"
        },
        "Coyote": {
            "scientific": "Canis latrans",
            "kingdom":    "Animalia",
            "phylum":     "Chordata",
            "class":      "Mammalia",
            "order":      "Carnivora",
            "family":     "Canidae",
            "genus":      "Canis"
        },
        ...
    }

Usage:
    conda activate birdclap
    python scripts/build_taxonomy_db.py
    python scripts/build_taxonomy_db.py --resume
"""

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import requests

# ── paths ─────────────────────────────────────────────────────────────────────
UNIFIED_META  = Path("data/xc_metadata_unified.csv")
CLASSES_CSV   = Path("data/classes.csv")
OUT_PATH      = Path("data/species_taxonomy.json")

GBIF_API   = "https://api.gbif.org/v1/species/match"
GBIF_DELAY = 0.25

# Highest-priority scientific name overrides — fixes GBIF incompatibilities
# (e.g. HawkEars uses newer Clements taxonomy names that GBIF doesn't map correctly)
SCI_NAME_CORRECTIONS: dict[str, str] = {
    "Cooper's Hawk":    "Accipiter cooperii",    # classes.csv has Astur cooperii → GBIF returns a plant
    "American Goshawk": "Accipiter gentilis",     # classes.csv has Astur atricapillus → wrong family
}

SKIP_LABELS = frozenset({
    "soundscape", "identity unknown", "noise", "speech", "canine", "squirrel",
    "insects", "rooster", "other", "background", "unknown", "nan", "",
})

# ── known scientific names for non-bird taxa ──────────────────────────────────
ADW_SCI_MAP: dict[str, str] = {
    "North American Red Squirrel": "Tamiasciurus hudsonicus",
    "Eastern Gray Squirrel":       "Sciurus carolinensis",
    "Eastern Chipmunk":            "Tamias striatus",
    "Big Brown Bat":               "Eptesicus fuscus",
    "Little Brown Myotis":         "Myotis lucifugus",
    "Northern Hoary Bat":          "Lasiurus cinereus",
    "Northern Myotis":             "Myotis septentrionalis",
    "Silver-haired Bat":           "Lasionycteris noctivagans",
    "Tricolored Bat":              "Perimyotis subflavus",
    "Eastern Red Bat":             "Lasiurus borealis",
    "Eastern Small-footed Myotis": "Myotis leibii",
    "Canadian Lynx":               "Lynx canadensis",
    "North American Porcupine":    "Erethizon dorsatum",
    "Coyote":                      "Canis latrans",
    "White-tailed Deer":           "Odocoileus virginianus",
    "Collared Pika":               "Ochotona collaris",
    "Singing Vole":                "Microtus miurus",
    "Taiga Vole":                  "Microtus gregalis",
    "Spring Peeper":               "Pseudacris crucifer",
    "Wood Frog":                   "Lithobates sylvaticus",
    "American Bullfrog":           "Lithobates catesbeianus",
    "American Toad":               "Anaxyrus americanus",
    "Canadian Toad":               "Anaxyrus hemiophrys",
    "Bronze Frog":                 "Lithobates clamitans",
    "Boreal Chorus Frog":          "Pseudacris maculata",
    "Eastern Gray Treefrog":       "Hyla versicolor",
    "Gray Treefrog":               "Hyla versicolor",
    "Green Frog":                  "Lithobates clamitans",
    "Northern Leopard Frog":       "Lithobates pipiens",
    "Pacific Chorus Frog":         "Pseudacris regilla",
    "Pickerel Frog":               "Lithobates palustris",
    "Great Basin Spadefoot":       "Spea intermontana",
    "Great Plains Toad":           "Anaxyrus cognatus",
    "Western Toad":                "Anaxyrus boreas",
    "Plains Spadefoot Toad":       "Spea bombifrons",
    "Steller's Sea Eagle":         "Haliaeetus pelagicus",
    "Broad-winged Bush-katydid":   "Scudderia pistillata",
    "Marsh ground cricket":        "Neoconocephalus palustris",
    "Riley's tree cricket":        "Oecanthus rileyi",
    "Steindachner's Shieldback":   "Neduba steindachneri",
    # Birds in unified metadata but not in classes.csv (no HawkEars code)
    "American Black Duck":         "Anas rubripes",
    "Arctic Redpoll":              "Acanthis hornemanni",
    "Black Guillemot":             "Cepphus grylle",
    "Black Scoter":                "Melanitta americana",
    "Black-bellied Whistling Duck":"Dendrocygna autumnalis",
    "Black-legged Kittiwake":      "Rissa tridactyla",
    "Black-necked Stilt":          "Himantopus mexicanus",
    "Burrowing Owl":               "Athene cunicularia",
    "Cackling Goose":              "Branta hutchinsii",
    "Calliope Hummingbird":        "Selasphorus calliope",
    "Canyon Wren":                 "Catherpes mexicanus",
    "Chinese Blackbird":           "Turdus mandarinus",
    "Citrine Wagtail":             "Motacilla citreola",
    "Common Eider":                "Somateria mollissima",
    "Common Redpoll":              "Acanthis flammea",
    "Cordilleran Flycatcher":      "Empidonax occidentalis",
    "Crested Myna":                "Acridotheres cristatellus",
    "Dusky Grouse":                "Dendragapus obscurus",
    "Eastern Yellow Wagtail":      "Motacilla tschutschensis",
    "Eurasian Skylark":            "Alauda arvensis",
    "Eurasian Wigeon":             "Mareca penelope",
    "Flammulated Owl":             "Psiloscops flammeolus",
    "Fork-tailed Storm Petrel":    "Hydrobates furcatus",
    "Glaucous Gull":               "Larus hyperboreus",
    "Greater Scaup":               "Aythya marila",
    "Grey-crowned Rosy Finch":     "Leucosticte tephrocotis",
    "Gyrfalcon":                   "Falco rusticolus",
    "Heermann's Gull":             "Larus heermanni",
    "Hermit Warbler":              "Setophaga occidentalis",
    "Iceland Gull":                "Larus glaucoides",
    "Leach's Storm Petrel":        "Hydrobates leucorhous",
    "Lesser Scaup":                "Aythya affinis",
    "Long-tailed Jaeger":          "Stercorarius longicaudus",
    "Northern Bobwhite":           "Colinus virginianus",
    "Northern Fulmar":             "Fulmarus glacialis",
    "Northern Gannet":             "Morus bassanus",
    "Northern Goshawk":            "Accipiter gentilis",
    "Northern Mockingbird":        "Mimus polyglottos",
    "Northern Wheatear":           "Oenanthe oenanthe",
    "Pacific-slope Flycatcher":    "Empidonax difficilis",
    "Parasitic Jaeger":            "Stercorarius parasiticus",
    "Pelagic Cormorant":           "Urile pelagicus",
    "Piping Plover":               "Charadrius melodus",
    "Pygmy Nuthatch":              "Sitta pygmaea",
    "Razorbill":                   "Alca torda",
    "Red Knot":                    "Calidris canutus",
    "Red Phalarope":               "Phalaropus fulicarius",
    "Red-breasted Merganser":      "Mergus serrator",
    "Red-throated Pipit":          "Anthus cervinus",
    "Roseate Tern":                "Sterna dougallii",
    "Scissor-tailed Flycatcher":   "Tyrannus forficatus",
    "Smith's Longspur":            "Calcarius pictus",
    "Sooty Grouse":                "Dendragapus fuliginosus",
    "Stilt Sandpiper":             "Calidris himantopus",
    "Surf Scoter":                 "Melanitta perspicillata",
    "Western Bluebird":            "Sialia mexicana",
    "Western Sandpiper":           "Calidris mauri",
    "White-throated Swift":        "Aeronautes saxatalis",
    "White-winged Scoter":         "Melanitta deglandi",
    "Yellow-browed Warbler":       "Phylloscopus inornatus",
}

# ── name aliases (variant → canonical) ───────────────────────────────────────
NAME_ALIASES: dict[str, str] = {
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
    "American Barn Owl":          "Barn Owl",
    "American Black Swift":       "Black Swift",
    "American Bushtit":           "Bushtit",
    "American Cliff Swallow":     "Cliff Swallow",
    "American Dusky Flycatcher":  "Dusky Flycatcher",
    "American Golden Plover":     "American Golden-Plover",
    "American Herring Gull":      "Herring Gull",
    "American Yellow Warbler":    "Yellow Warbler",
    "Arctic Redpoll":             "Common Redpoll",
    "Audubon's Warbler":          "Yellow-rumped Warbler",
    "Sand Martin":                "Bank Swallow",
    "Two-barred Crossbill":       "White-winged Crossbill",
    "Common Starling":            "European Starling",
    "European Herring Gull":      "Herring Gull",
    "Common Pheasant":            "Ring-necked Pheasant",
    "Brant Goose":                "Brant",
    "Buff-bellied Pipit":         "American Pipit",
    "Black-necked Grebe":         "Eared Grebe",
    "Red Fox Sparrow":            "Fox Sparrow",
    "Slate-colored Fox Sparrow":  "Fox Sparrow",
    "Sooty Fox Sparrow":          "Fox Sparrow",
    "Myrtle Warbler":             "Yellow-rumped Warbler",
    "Northern Pygmy Owl":         "Northern Pygmy-Owl",
    "Western Screech Owl":        "Western Screech-Owl",
    "Eastern Screech Owl":        "Eastern Screech-Owl",
    "Northern Hawk-Owl":          "Northern Hawk Owl",
    "Eastern Wood Pewee":         "Eastern Wood-Pewee",
    "Western Wood Pewee":         "Western Wood-Pewee",
    "Pacific-slope Flycatcher":   "Pacific-Slope Flycatcher",
    "Eurasian Collared Dove":     "Eurasian Collared-Dove",
    "Leach's Storm Petrel":       "Leach's Storm-Petrel",
    "Rock Dove":                  "Rock Pigeon",
    "Eastern Osprey":             "Osprey",
    "Western Osprey":             "Osprey",
    "Northern Raven":             "Common Raven",
    "Hudsonian Whimbrel":         "Whimbrel",
    "Redwing":                    "American Robin",
}


# ── GBIF lookup ───────────────────────────────────────────────────────────────

def gbif_lookup(scientific_name: str) -> dict:
    """Query GBIF species match. Returns dict with taxonomy fields."""
    try:
        r = requests.get(
            GBIF_API,
            params={"name": scientific_name, "strict": "false", "verbose": "false"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "kingdom": d.get("kingdom", ""),
            "phylum":  d.get("phylum",  ""),
            "class":   d.get("class",   ""),
            "order":   d.get("order",   ""),
            "family":  d.get("family",  ""),
            "genus":   d.get("genus",   ""),
        }
    except Exception as e:
        print(f"    [GBIF error] {scientific_name}: {e}")
        return {}


def gbif_search_common(common_name: str) -> tuple[str, dict]:
    """Search GBIF by common name. Returns (scientific_name, taxonomy_dict)."""
    try:
        r = requests.get(
            "https://api.gbif.org/v1/species/suggest",
            params={"q": common_name, "limit": 3},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        for res in results:
            sci = res.get("scientificName", "").split(" (")[0].strip()
            # Clean out authorship
            parts = sci.split()
            if len(parts) >= 2:
                sci_clean = f"{parts[0]} {parts[1]}"
                tax = gbif_lookup(sci_clean)
                if tax.get("order"):
                    return sci_clean, tax
        return "", {}
    except Exception:
        return "", {}


# ── scientific name resolution ────────────────────────────────────────────────

def resolve_scientific_name(common_name: str,
                             classes_map: dict[str, str]) -> str:
    """
    Return best scientific name for a common name.
    Priority: SCI_NAME_CORRECTIONS → classes.csv → ADW_SCI_MAP → alias → empty.
    """
    # Highest priority: manual corrections for GBIF-incompatible names
    if common_name in SCI_NAME_CORRECTIONS:
        return SCI_NAME_CORRECTIONS[common_name]

    # Direct hit in classes.csv
    if common_name in classes_map and classes_map[common_name]:
        return classes_map[common_name]

    # Direct hit in ADW overrides
    if common_name in ADW_SCI_MAP:
        return ADW_SCI_MAP[common_name]

    # Resolve alias then look up
    alias = NAME_ALIASES.get(common_name)
    if alias and alias != common_name:
        if alias in classes_map and classes_map[alias]:
            return classes_map[alias]
        if alias in ADW_SCI_MAP:
            return ADW_SCI_MAP[alias]

    return ""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+", default=[str(UNIFIED_META)])
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--resume", action="store_true",
                        help="Skip species already in output with non-empty order")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if resuming
    existing: dict = {}
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        done = sum(1 for v in existing.values() if v.get("order"))
        print(f"Resuming — {done}/{len(existing)} entries with order")

    # Load classes.csv scientific name map
    classes_map: dict[str, str] = {}
    if CLASSES_CSV.exists():
        with open(CLASSES_CSV) as f:
            for row in csv.DictReader(f):
                n = row.get("Name", "").strip()
                s = row.get("AltName", "").strip()
                if n and s:
                    classes_map[n] = s

    # Collect all unique species from metadata
    all_names: set[str] = set()
    for path in args.metadata:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        name_col = next((c for c in ("common_name", "species", "name") if c in df.columns), None)
        if name_col:
            all_names.update(df[name_col].dropna().astype(str).str.strip().tolist())

    # Also include all classes.csv birds
    all_names.update(classes_map.keys())

    # Filter skip labels
    species_list = sorted(n for n in all_names if n.lower() not in SKIP_LABELS and n)
    print(f"Total species: {len(species_list)}")

    results = dict(existing)
    counts  = {"resolved": 0, "gbif_search": 0, "no_sci": 0, "no_tax": 0}

    for i, name in enumerate(species_list, 1):
        if args.resume and name in results and results[name].get("order"):
            continue

        sci = resolve_scientific_name(name, classes_map)

        if not sci:
            # Last resort: GBIF common name search
            sci, tax = gbif_search_common(name)
            time.sleep(GBIF_DELAY)
            if sci and tax.get("order"):
                results[name] = {"scientific": sci, **tax}
                counts["gbif_search"] += 1
                print(f"  [{i:4d}/{len(species_list)}] {name}: {sci} [GBIF search]")
            else:
                results[name] = {"scientific": "", "kingdom": "", "phylum": "",
                                 "class": "", "order": "", "family": "", "genus": ""}
                counts["no_sci"] += 1
                print(f"  [{i:4d}/{len(species_list)}] {name}: no scientific name found")
            continue

        # GBIF taxonomy lookup by scientific name
        tax = gbif_lookup(sci)
        time.sleep(GBIF_DELAY)

        if tax.get("order"):
            results[name] = {"scientific": sci, **tax}
            counts["resolved"] += 1
            chain = f"{tax.get('class','')} > {tax.get('order','')} > {tax.get('family','')}"
            print(f"  [{i:4d}/{len(species_list)}] {name}: {chain}")
        else:
            results[name] = {"scientific": sci, "kingdom": "", "phylum": "",
                             "class": "", "order": "", "family": "", "genus": ""}
            counts["no_tax"] += 1
            print(f"  [{i:4d}/{len(species_list)}] {name}: {sci} — no taxonomy from GBIF")

        if i % 20 == 0:
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'─'*60}")
    print(f"Done. {len(results)} species → {out_path}")
    print(f"  Full taxonomy  : {counts['resolved']}")
    print(f"  GBIF name srch : {counts['gbif_search']}")
    print(f"  No sci name    : {counts['no_sci']}")
    print(f"  No GBIF result : {counts['no_tax']}")

    # Show class breakdown
    from collections import Counter
    classes_count = Counter(v.get("class", "unknown") or "unknown"
                            for v in results.values())
    print(f"\nClass breakdown:")
    for cls, cnt in classes_count.most_common():
        print(f"  {cls or 'unknown':<20}: {cnt}")


if __name__ == "__main__":
    main()
