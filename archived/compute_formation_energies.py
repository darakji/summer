#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute formation energies (per atom) for all .cif files in a folder using CHGNet and MACE.

Usage:
  python compute_formation_energies.py \
      --folder ./formntest \
      --mace-model /path/to/mace_mp_trj_universal.model \
      --device cuda

Notes:
- Single-point energies (no relaxation).
- Uses user-supplied elemental chemical potentials (per atom) for both models.
- Outputs Excel (.xlsx) and JSON (.json) with eV/atom and meV/atom columns.
- Requires: ase, chgnet, mace, pandas, openpyxl (for Excel).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
from ase.io import read

# --- User-provided elemental chemical potentials (eV/atom) ---
MU_CHGNET: Dict[str, float] = {"Li": -1.882, "La": -4.894, "Zr": -8.509, "O": -4.913}

MU_MACE: Dict[str, float] = {
    "Li": -1.894110,
    "La": -4.891070,
    "Zr": -8.539969,
    "O":  -4.870034,
}

SUPPORTED_ELEMENTS = sorted(set(MU_CHGNET) | set(MU_MACE))


def get_composition(atoms) -> Counter:
    return Counter(atoms.get_chemical_symbols())


def formula_from_comp(comp: Counter) -> str:
    # Simple pretty formula like Li7La3Zr2O12 (order by alphabetical by default)
    parts = []
    for el in sorted(comp.keys()):
        n = comp[el]
        parts.append(f"{el}{'' if n == 1 else n}")
    return "".join(parts)


def formation_energy_per_atom(E_tot: float, comp: Counter, mu: Dict[str, float]) -> float:
    """(E_tot - sum_i n_i mu_i) / N"""
    # Validate all elements present in mu
    missing = [el for el in comp if el not in mu]
    if missing:
        raise ValueError(f"Missing elemental μ for: {missing}")
    N = sum(comp.values())
    sum_ref = sum(n * mu[el] for el, n in comp.items())
    return (E_tot - sum_ref) / N


def compute_energies_for_file(
    cif_path: Path,
    chgnet_calc,
    mace_calc,
) -> Dict[str, Any]:
    """Return dict of results for a single CIF; errors captured in 'error' field."""
    out = {
        "file": str(cif_path),
        "n_atoms": None,
        "formula": None,
        "E_tot_CHGNet_eV": None,
        "E_form_CHGNet_eV_per_atom": None,
        "E_form_CHGNet_meV_per_atom": None,
        "E_tot_MACE_eV": None,
        "E_form_MACE_eV_per_atom": None,
        "E_form_MACE_meV_per_atom": None,
        "error": None,
    }
    try:
        atoms = read(str(cif_path))
        comp = get_composition(atoms)
        out["n_atoms"] = len(atoms)
        out["formula"] = formula_from_comp(comp)
    except Exception as e:
        out["error"] = f"Failed to read CIF: {e}"
        return out

    # CHGNet single-point
    try:
        if chgnet_calc is not None:
            atoms_chg = atoms.copy()
            atoms_chg.calc = chgnet_calc
            E_tot_chg = atoms_chg.get_potential_energy()  # eV total
            Ef_chg = formation_energy_per_atom(E_tot_chg, comp, MU_CHGNET)
            out["E_tot_CHGNet_eV"] = float(E_tot_chg)
            out["E_form_CHGNet_eV_per_atom"] = float(Ef_chg)
            out["E_form_CHGNet_meV_per_atom"] = float(Ef_chg * 1000.0)
    except Exception as e:
        out["error"] = f"(CHGNet) {e}"

    # MACE single-point
    try:
        if mace_calc is not None:
            atoms_mc = atoms.copy()
            atoms_mc.calc = mace_calc
            E_tot_mc = atoms_mc.get_potential_energy()  # eV total
            Ef_mc = formation_energy_per_atom(E_tot_mc, comp, MU_MACE)
            out["E_tot_MACE_eV"] = float(E_tot_mc)
            out["E_form_MACE_eV_per_atom"] = float(Ef_mc)
            out["E_form_MACE_meV_per_atom"] = float(Ef_mc * 1000.0)
    except Exception as e:
        # append error info, but keep CHGNet results if present
        prev = out["error"] + " | " if out["error"] else ""
        out["error"] = prev + f"(MACE) {e}"

    return out


def main():
    parser = argparse.ArgumentParser(description="Formation energies for CIFs using CHGNet and MACE.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .cif files")
    parser.add_argument("--mace-model", type=str, required=True, help="Path to MACE model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device for MACE")
    parser.add_argument("--head", type=str, default=None, help="Optional head name for multi-head MACE models")
    parser.add_argument("--outfile", type=str, default="formation_energies_summary", help="Base name for outputs")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    cif_files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".cif"])
    if not cif_files:
        print(f"[ERROR] No .cif files found in {folder}", file=sys.stderr)
        sys.exit(3)

    # --- Instantiate calculators ---
    # CHGNet (ASE)
    try:
        # User prefers importing CHGNetCalculator from chgnet.model.dynamics
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        chg_model = CHGNet.load()  # universal
        chgnet_calc = CHGNetCalculator(model=chg_model)
    except Exception as e:
        print(f"[WARN] CHGNet unavailable: {e}. CHGNet results will be empty.", file=sys.stderr)
        chgnet_calc = None

    # MACE (ASE)
    try:
        from mace.calculators import MACECalculator
        mace_calc = MACECalculator(
            model_path=args.mace_model,
            device=args.device,
            head=args.head,
        )
    except Exception as e:
        print(f"[WARN] MACE unavailable: {e}. MACE results will be empty.", file=sys.stderr)
        mace_calc = None

    # --- Process files ---
    results: List[Dict[str, Any]] = []
    for i, cif in enumerate(cif_files, 1):
        print(f"[{i}/{len(cif_files)}] {cif.name}")
        res = compute_energies_for_file(cif, chgnet_calc, mace_calc)
        results.append(res)

    df = pd.DataFrame(results)

    # Save JSON
    json_path = Path(f"{args.outfile}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "folder": str(folder.resolve()),
                "supported_elements": SUPPORTED_ELEMENTS,
                "mu_chgnet_eV_per_atom": MU_CHGNET,
                "mu_mace_eV_per_atom": MU_MACE,
                "notes": "Energies are single-point totals (no relaxation). Formation energy per atom: (E_tot - sum_i n_i mu_i) / N.",
            },
            "results": results
        }, f, indent=2)
    print(f"[OK] Wrote {json_path}")

    # Save Excel
    xlsx_path = Path(f"{args.outfile}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="summary")
        # Write μ tables on separate sheets
        pd.DataFrame([MU_CHGNET]).to_excel(writer, index=False, sheet_name="mu_CHGNet")
        pd.DataFrame([MU_MACE]).to_excel(writer, index=False, sheet_name="mu_MACE")
    print(f"[OK] Wrote {xlsx_path}")

    # Quick CLI view of failures
    n_err = df["error"].notna().sum()
    if n_err:
        print(f"[WARN] {n_err} file(s) had errors. See 'error' column in outputs.")
    else:
        print("[OK] All files processed without reported errors.")

if __name__ == "__main__":
    main()
