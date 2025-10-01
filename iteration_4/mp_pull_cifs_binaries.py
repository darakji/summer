#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pull binaries, ternaries, quaternaries in {La, Li, Zr, O} from Materials Project.
- Uses elements + nelements (no 'fields' param).
- Filters energy_above_hull client-side (stable-only or within EHULL_MAX).
- Picks ONE 'best' structure per reduced formula (customizable).
- Saves CIFs under OUTDIR/{2-ary,3-ary,4-ary}/
- Writes mp_pull_manifest.json and .xlsx with metadata.

Usage:
  - Put your MP API key below or in the environment as MAPI_KEY.
"""

import os
import json
import time
import math
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# ------------------ CONFIG ------------------
API_KEY = "j3J85pX4nLw6asHG9E2lbbCHEKDKgrjc"  # set here or via env
ELEMENTS = ["La", "Li", "Zr", "O"]                  # base element set
R_VALUES = [1, 2, 3, 4]                                # binaries, ternaries, quaternaries

OUTDIR = "mp_pull_la_li_zr_o"                       # output root
INCLUDE_METASTABLE = True                           # True keeps entries within EHULL_MAX
EHULL_MAX = 0.10                                    # eV/atom threshold for metastability
MAX_PER_FORMULA = 1                                 # number to keep per reduced formula
PRIORITIZE_LOW_EHULL_OVER_EATOM = True              # tie-breaker preference
ALLOW_DEPRECATED = False                            # skip deprecated entries by default
MAX_RETRIES = 4                                     # API retries with backoff
# --------------------------------------------


def safe_get(d, *keys, default=None):
    """Nested dict-safe getter: safe_get(x, 'a', 'b', default=None)."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def backoff_sleep(k):
    """Exponential backoff sleep (seconds)."""
    time.sleep(min(2 ** k, 10))


def mp_search_combo(mpr: MPRester, combo, include_metastable=True, ehull_max=0.1):
    """
    Query MP Summary docs for exactly the elements in 'combo', robustly.
    Avoids passing 'fields' to be compatible with servers that reject it.
    Applies server-side filter only where widely supported; otherwise
    we will still filter client-side.
    """
    kwargs = dict(elements=list(combo), nelements=len(combo))
    # Try adding energy_above_hull_max (some deployments accept it)
    if include_metastable:
        kwargs["energy_above_hull_max"] = ehull_max
    else:
        kwargs["energy_above_hull_max"] = 0.0

    # Retry with backoff; if the parameter causes issues, fall back.
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return mpr.summary.search(**kwargs)
        except Exception as e:
            last_err = e
            # Fallback: remove potential problematic filter and try again
            if "energy_above_hull_max" in kwargs:
                del kwargs["energy_above_hull_max"]
            backoff_sleep(attempt)
    # Final fallback (no server filters)
    try:
        return mpr.summary.search(elements=list(combo), nelements=len(combo))
    except Exception as e2:
        raise RuntimeError(f"MP search failed for {combo}: {e2}. Last error: {last_err}") from e2


def choose_best_per_formula(rows):
    """
    For a list of MP docs with the same reduced formula, pick best entries:
      1) Prefer stable (is_stable True or E_hull ~ 0)
      2) Else minimal energy_above_hull
      3) Tie-break by minimal energy_per_atom
    Returns up to MAX_PER_FORMULA rows.
    """
    if not rows:
        return []

    def e_hull(r):
        # Some payloads use 'energy_above_hull', ensure float
        val = r.get("energy_above_hull")
        try:
            return float(val) if val is not None else math.inf
        except Exception:
            return math.inf

    def e_atom(r):
        val = r.get("energy_per_atom")
        try:
            return float(val) if val is not None else math.inf
        except Exception:
            return math.inf

    # Stable if is_stable True OR e_hull ~ 0
    stable = [r for r in rows if bool(r.get("is_stable")) or (e_hull(r) <= 1e-8)]
    pool = stable if stable else rows

    if PRIORITIZE_LOW_EHULL_OVER_EATOM:
        pool = sorted(pool, key=lambda r: (e_hull(r), e_atom(r)))
    else:
        pool = sorted(pool, key=lambda r: (e_atom(r), e_hull(r)))

    return pool[:MAX_PER_FORMULA]


def save_cif_and_record(rec, bucket_dir, seen_paths):
    """
    Save a CIF for a single MP doc. Returns saved relative path.
    Handles collisions by appending _1, _2, ...
    """
    # prefer pretty formula; fall back to anonymous
    formula = rec.get("formula_pretty") or rec.get("formula_anonymous") or "UNKNOWN"
    mpid = rec.get("material_id", "noid")
    tag = f"{formula}__{mpid}"
    path = os.path.join(bucket_dir, f"{tag}.cif")
    base = path
    idx = 1
    while path in seen_paths or os.path.exists(path):
        path = base.replace(".cif", f"_{idx}.cif")
        idx += 1

    structure = rec.get("structure")
    if isinstance(structure, Structure):
        structure.to(fmt="cif", filename=path)
    else:
        # Some payloads include dict-like structure; try reconstruct
        try:
            Structure.from_dict(structure).to(fmt="cif", filename=path)
        except Exception as e:
            raise RuntimeError(f"Could not write CIF for {mpid} ({formula}): {e}")

    seen_paths.add(path)
    return path


def main():
    if not API_KEY or API_KEY == "YOUR_MP_API_KEY":
        raise SystemExit("Please set API_KEY (env MAPI_KEY or edit the script).")

    os.makedirs(OUTDIR, exist_ok=True)
    all_rows = []
    seen_paths = set()

    with MPRester(API_KEY) as mpr:
        for r in R_VALUES:
            combos = list(combinations(sorted(ELEMENTS), r))
            for combo in combos:
                combo_str = "-".join(combo)
                print(f"[INFO] Querying combo={combo_str} (r={r}) ...")
                try:
                    rows = mp_search_combo(
                        mpr,
                        combo=combo,
                        include_metastable=INCLUDE_METASTABLE,
                        ehull_max=EHULL_MAX,
                    )
                except Exception as e:
                    print(f"[WARN] Search failed for {combo_str}: {e}")
                    rows = []

                # Optional: skip deprecated
                if not ALLOW_DEPRECATED:
                    rows = [x for x in rows if not bool(x.get("deprecated"))]

                # Client-side filter on E_hull (in case server ignored it)
                def pass_ehull(x):
                    eah = x.get("energy_above_hull")
                    try:
                        eah = float(eah) if eah is not None else math.inf
                    except Exception:
                        eah = math.inf
                    if INCLUDE_METASTABLE:
                        return eah <= EHULL_MAX
                    else:
                        return eah <= 1e-8

                rows = [x for x in rows if pass_ehull(x)]

                # Group by reduced formula (formula_pretty ~ reduced)
                by_formula = defaultdict(list)
                for doc in rows:
                    f = doc.get("formula_pretty") or doc.get("formula_anonymous") or "UNKNOWN"
                    by_formula[f].append(doc)

                picked = []
                for f, group in by_formula.items():
                    picked.extend(choose_best_per_formula(group))

                # Save CIFs and collect metadata
                bucket = os.path.join(OUTDIR, f"{r}-ary")
                os.makedirs(bucket, exist_ok=True)

                for rec in picked:
                    try:
                        cif_path = save_cif_and_record(rec, bucket, seen_paths)
                    except Exception as e:
                        print(f"[WARN] CIF save failed for {rec.get('material_id')}: {e}")
                        continue

                    # Extract metadata robustly
                    mpid = rec.get("material_id")
                    formula = rec.get("formula_pretty") or rec.get("formula_anonymous")
                    epa = rec.get("energy_per_atom")
                    eah = rec.get("energy_above_hull")
                    fea = rec.get("formation_energy_per_atom")
                    density = rec.get("density")
                    volume = rec.get("volume")
                    is_stable = bool(rec.get("is_stable"))
                    deprecated = bool(rec.get("deprecated"))
                    # space group can be nested
                    sg = (
                        safe_get(rec, "spacegroup", "symbol")
                        or safe_get(rec, "symmetry", "symbol")
                        or rec.get("spacegroup.symbol")
                        or rec.get("symmetry.symbol")
                    )

                    all_rows.append({
                        "r": r,
                        "elements": combo_str,
                        "formula": formula,
                        "material_id": mpid,
                        "is_stable": is_stable,
                        "E_per_atom_eV": epa,
                        "E_above_hull_eV_per_atom": eah,
                        "E_form_per_atom_eV": fea,
                        "spacegroup": sg,
                        "volume_A3": volume,
                        "density_g_cm3": density,
                        "deprecated": deprecated,
                        "cif_path": os.path.relpath(cif_path, OUTDIR),
                    })

    # Write manifests
    manifest_json = os.path.join(OUTDIR, "mp_pull_manifest.json")
    with open(manifest_json, "w") as f:
        json.dump(all_rows, f, indent=2)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["r", "elements", "formula", "material_id"], kind="stable")
    manifest_xlsx = os.path.join(OUTDIR, "mp_pull_manifest.xlsx")
    with pd.ExcelWriter(manifest_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="pull")

        # Add a small summary sheet
        if not df.empty:
            summary = (
                df.groupby(["r"])
                  .size()
                  .rename("count")
                  .reset_index()
                  .sort_values("r")
            )
            summary.to_excel(writer, index=False, sheet_name="summary")

    # Console summary
    if df.empty:
        print("\n=== DONE (no results) ===")
    else:
        counts = df.groupby("r").size().to_dict()
        print("\n=== DONE ===")
        print(f"Saved CIFs under: {OUTDIR}/(2-ary, 3-ary, 4-ary)")
        print(f"Manifest JSON  : {manifest_json}")
        print(f"Manifest Excel : {manifest_xlsx}")
        print(f"Counts per r   : {counts}")


if __name__ == "__main__":
    main()
