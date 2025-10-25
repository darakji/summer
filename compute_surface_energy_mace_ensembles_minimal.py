#!/usr/bin/env python3
import os, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read
from ase.atoms import Atoms

# ===== Constants =====
LLZO_FORMULA = {"Li": 7, "La": 3, "Zr": 2, "O": 12}
LLZO_ELEMENTS = set(LLZO_FORMULA.keys())
EV_TO_J = 1.602176634e-19
A2_TO_M2 = 1e-20

# ===== Helpers =====
def count_elements(at: Atoms) -> dict:
    from collections import Counter
    return dict(Counter(at.get_chemical_symbols()))

def formula_units_in_cell(counts: dict, base_formula: dict) -> int:
    ratios = [(counts.get(el, 0) / req) for el, req in base_formula.items()]
    n = min(ratios)
    n_int = int(round(n))
    if not (abs(n - n_int) < 1e-6):
        raise ValueError(f"Non-integer f.u. detected (n={n}); slab may not be stoichiometric. Counts={counts}")
    return n_int

def parse_facet_and_term(path: str):
    p = Path(path)
    facet, term = None, None
    for s in p.parts:
        if "_terminated_" in s:
            term = s.split("_terminated_")[0] + "_terminated"
            facet = s.split("_terminated_")[1]
            break
    return facet, term

def auto_inplane_area(at: Atoms):
    """Robust fallback if facet not parseable/aligned."""
    cell = at.get_cell()
    pbc = at.get_pbc()

    def _axis_dir(i):
        v = np.array(cell[i], dtype=float)
        n = np.linalg.norm(v)
        return v / n, n

    def atomic_span_along_axis(axis_i: int):
        dir_i, L = _axis_dir(axis_i)
        pos = at.get_positions()
        projs = pos @ dir_i
        return float(projs.max() - projs.min()), float(L)

    # 1) pbc-based
    if isinstance(pbc, (list, tuple, np.ndarray)) and len(pbc) == 3:
        false_axes = [i for i, flag in enumerate(pbc) if not bool(flag)]
        if len(false_axes) == 1:
            vac = false_axes[0]
            inplane = [0, 1, 2]; inplane.remove(vac)
            area = np.linalg.norm(np.cross(cell[inplane[0]], cell[inplane[1]]))
            return float(area), vac, tuple(inplane), "pbc"

    # 2) geometric heuristic
    vf, lengths = [], []
    for i in range(3):
        span, L = atomic_span_along_axis(i)
        lengths.append(L)
        vf.append(max(0.0, 1.0 - span / max(L, 1e-12)))
    vac = int(np.lexsort((lengths, vf))[-1])  # max vacuum fraction; tie by longest L
    inplane = [0, 1, 2]; inplane.remove(vac)
    area = np.linalg.norm(np.cross(cell[inplane[0]], cell[inplane[1]]))
    return float(area), vac, tuple(inplane), "vacuum_fraction"

def area_from_facet(slab_path: str, at: Atoms):
    """Facet-map for (001)/(010)/(100); fallback for others."""
    facet, _ = parse_facet_and_term(slab_path)
    cell = at.get_cell()

    def A(idx0, idx1, vac):
        return float(np.linalg.norm(np.cross(cell[idx0], cell[idx1]))), vac, (idx0, idx1), "facet_map"

    if facet == "001": return A(0, 1, 2)
    if facet == "010": return A(0, 2, 1)
    if facet == "100": return A(1, 2, 0)
    return auto_inplane_area(at)

def get_mace_energy(at: Atoms, model_path: str, device: str, head: str = "target_head") -> float:
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_path=model_path, device=device, head=head)
    at.calc = calc
    return float(at.get_potential_energy())

def evaluate_ensemble(ensemble_name: str, model_paths: list, device: str,
                      bulk_path: str, stoich_dir: str) -> pd.DataFrame:
    """Return a single DataFrame with m1..m4 and avg columns for each slab."""
    # Bulk (per model)
    bulk = read(bulk_path)
    bulk_counts = count_elements(bulk)
    bulk_fu = formula_units_in_cell(bulk_counts, LLZO_FORMULA)
    n_atoms_bulk = len(bulk)

    E_bulk_models = []
    E_bulk_per_fu_models = []
    for mp in model_paths:
        E_bulk = get_mace_energy(bulk.copy(), mp, device=device, head="target_head")
        E_bulk_models.append(E_bulk)
        E_bulk_per_fu_models.append(E_bulk / bulk_fu)

    # Slabs
    rows = []
    slab_files = sorted(glob.glob(os.path.join(stoich_dir, "**", "*.vasp"), recursive=True))

    for fpath in slab_files:
        slab = read(fpath)
        counts = count_elements(slab)
        if set(counts) - LLZO_ELEMENTS:
            continue

        n_fu = formula_units_in_cell(counts, LLZO_FORMULA)
        n_atoms = len(slab)
        area_A2, vac_axis, inplane_axes, method = area_from_facet(fpath, slab)

        # Per-model values
        E_slab = []
        gamma_eV_A2 = []
        gamma_J_m2  = []
        dE_pa = []  # delta E per atom

        for k, mp in enumerate(model_paths):
            Es = get_mace_energy(slab.copy(), mp, device=device, head="target_head")
            E_slab.append(Es)
            g_eva2 = (Es - n_fu * E_bulk_per_fu_models[k]) / (2.0 * area_A2)
            gamma_eV_A2.append(g_eva2)
            gamma_J_m2.append(g_eva2 * (EV_TO_J / A2_TO_M2))
            dE_pa.append((Es - E_bulk_models[k]) / n_atoms)

        # Averages
        Es_avg = float(np.mean(E_slab))
        g_eva2_avg = float(np.mean(gamma_eV_A2))
        g_Jm2_avg  = float(np.mean(gamma_J_m2))
        dE_pa_avg  = float(np.mean(dE_pa))

        facet, term = parse_facet_and_term(fpath)
        row = {
            "ensemble": ensemble_name,
            "file": fpath,
            "facet": facet,
            "termination": term,
            "n_formula_units": n_fu,
            "n_atoms": n_atoms,
            "area_A2": area_A2,
            "vacuum_axis": vac_axis,
            "inplane_axes": f"{inplane_axes}",
            "area_detection": method,
            # averages
            "E_slab_eV_avg": Es_avg,
            "gamma_eV_A2_avg": g_eva2_avg,
            "gamma_J_m2_avg": g_Jm2_avg,
            "deltaE_eV_per_atom_avg": dE_pa_avg,
        }
        # per-model columns (m1..m4)
        for k, mp in enumerate(model_paths, start=1):
            row[f"E_slab_eV_m{k}"] = E_slab[k-1]
            row[f"gamma_eV_A2_m{k}"] = gamma_eV_A2[k-1]
            row[f"gamma_J_m2_m{k}"] = gamma_J_m2[k-1]
            row[f"deltaE_eV_per_atom_m{k}"] = dE_pa[k-1]
            row[f"model_path_m{k}"] = mp

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["facet", "termination", "gamma_J_m2_avg", "file"], na_position="last"
    ).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser("Minimal MACE ensembles surface energy")
    ap.add_argument("--base", default="/home/phanim/harshitrawat/summer/MS_LLZO_surface_data-master")
    ap.add_argument("--bulk", default=None)
    ap.add_argument("--stoich_dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_xlsx", default="/home/phanim/harshitrawat/summer/surface_energy_MACE_ensembles.xlsx")
    ap.add_argument("--out_json", default="/home/phanim/harshitrawat/summer/surface_energy_MACE_ensembles.json")
    args = ap.parse_args()

    BASE = args.base
    BULK = args.bulk or os.path.join(BASE, "refs", "LLZO_tet_bulk.cif")
    STOICH = args.stoich_dir or os.path.join(BASE, "stoichiometric")

    MACE_T2 = [
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T2_w1_2heads_final_run-10.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T2_w2_2heads_final_run-20.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T2_w3_2heads_final_run-30.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T2_w4_2heads_final_run-40.model",
    ]
    MACE_T1 = [
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T1_w1_2heads_final_run-10.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T1_w2_2heads_final_run-20.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T1_w3_2heads_final_run-40.model",
        "/home/phanim/harshitrawat/summer/checkpoints/mace_T1_w4_2heads_final_run-30.model",
    ]

    df_t2 = evaluate_ensemble("MACET2", MACE_T2, args.device, BULK, STOICH)
    df_t1 = evaluate_ensemble("MACET1", MACE_T1, args.device, BULK, STOICH)

    # Write ONE excel with TWO sheets
    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as xw:
        df_t2.to_excel(xw, sheet_name="MACET2", index=False)
        df_t1.to_excel(xw, sheet_name="MACET1", index=False)

    # One JSON containing both
    out_json_obj = {"MACET2": df_t2.to_dict(orient="records"),
                    "MACET1": df_t1.to_dict(orient="records")}
    with open(args.out_json, "w") as f:
        json.dump(out_json_obj, f, indent=2)

    print(f"Wrote:\n  {args.out_xlsx}\n  {args.out_json}")

if __name__ == "__main__":
    main()
