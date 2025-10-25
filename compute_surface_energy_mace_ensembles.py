#!/usr/bin/env python3
import os, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read
from ase.atoms import Atoms

# ===================== CONSTANTS =====================
LLZO_FORMULA = {"Li": 7, "La": 3, "Zr": 2, "O": 12}
LLZO_ELEMENTS = set(LLZO_FORMULA.keys())
EV_TO_J = 1.602176634e-19
A2_TO_M2 = 1e-20

# ===================== HELPERS =====================
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
    """Robust fallback if facet not parseable."""
    cell = at.get_cell()
    pbc = at.get_pbc()

    def _axis_dir(i):
        v = np.array(cell[i], dtype=float)
        n = np.linalg.norm(v)
        return v / n, n

    def atomic_span_along_axis(axis_i: int):
        dir_i, length_i = _axis_dir(axis_i)
        pos = at.get_positions()
        projs = pos @ dir_i
        return float(projs.max() - projs.min()), float(length_i)

    # 1) pbc-based
    if isinstance(pbc, (list, tuple, np.ndarray)) and len(pbc) == 3:
        false_axes = [i for i, flag in enumerate(pbc) if not bool(flag)]
        if len(false_axes) == 1:
            vac = false_axes[0]
            inplane = [0, 1, 2]
            inplane.remove(vac)
            area = np.linalg.norm(np.cross(cell[inplane[0]], cell[inplane[1]]))
            return float(area), vac, tuple(inplane), "pbc"

    # 2) geometric heuristic
    vf = []
    lengths = []
    for i in range(3):
        span, L = atomic_span_along_axis(i)
        lengths.append(L)
        vf.append(max(0.0, 1.0 - span / max(L, 1e-12)))
    vac = int(np.lexsort((lengths, vf))[-1])  # max vacuum fraction; tie-break by longest axis
    inplane = [0, 1, 2]; inplane.remove(vac)
    area = np.linalg.norm(np.cross(cell[inplane[0]], cell[inplane[1]]))
    return float(area), vac, tuple(inplane), "vacuum_fraction"

def area_from_facet(slab_path: str, at: Atoms):
    """
    Paper-faithful area selection for (001)/(010)/(100);
    fallback to robust auto-detect for off-axis facets like (110)/(101)/(011)/(111)/(021)/(031).
    Returns (area_A2, vac_axis, inplane_axes, method)
    """
    facet, _ = parse_facet_and_term(slab_path)
    cell = at.get_cell()

    def area_for_inplane(idx0, idx1, vac):
        A = float(np.linalg.norm(np.cross(cell[idx0], cell[idx1])))
        return A, vac, (idx0, idx1), "facet_map"

    if facet in {"001", "010", "100"}:
        if facet == "001":   # normal ~ c -> in-plane a,b
            return area_for_inplane(0, 1, 2)
        if facet == "010":   # normal ~ b -> in-plane a,c
            return area_for_inplane(0, 2, 1)
        if facet == "100":   # normal ~ a -> in-plane b,c
            return area_for_inplane(1, 2, 0)

    # Off-axis (or facet missing) -> fallback
    return auto_inplane_area(at)

def get_mace_energy(at: Atoms, model_path: str, device: str, head: str = "target_head") -> float:
    """Return energy (eV) from a specific MACE model."""
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_path=model_path, device=device, head=head)
    at.calc = calc
    return float(at.get_potential_energy())

def eval_ensemble(base_dir: str,
                  bulk_path: str,
                  stoich_dir: str,
                  model_paths: list,
                  device: str,
                  ensemble_name: str,
                  out_prefix: str):
    """
    Evaluate surface energies with an ensemble of MACE models.
    Writes detailed and summary Excel/JSON files with ensemble averages.
    """
    # === Bulk reference per model ===
    bulk = read(bulk_path)
    bulk_counts = count_elements(bulk)
    bulk_fu = formula_units_in_cell(bulk_counts, LLZO_FORMULA)
    n_atoms_bulk = len(bulk)

    E_bulk_models = []
    for mp in model_paths:
        E_bulk = get_mace_energy(bulk.copy(), mp, device=device, head="target_head")
        E_bulk_models.append(E_bulk)

    E_bulk_per_fu_models = [E / bulk_fu for E in E_bulk_models]
    E_bulk_per_atom_models = [E / n_atoms_bulk for E in E_bulk_models]

    # === Slabs ===
    rows_detail = []
    slab_files = sorted(glob.glob(os.path.join(stoich_dir, "**", "*.vasp"), recursive=True))

    for fpath in slab_files:
        slab = read(fpath)
        counts = count_elements(slab)
        if set(counts) - LLZO_ELEMENTS:
            # foreign species -> skip
            continue
        n_fu = formula_units_in_cell(counts, LLZO_FORMULA)
        n_atoms = len(slab)

        # area
        area_A2, vac_axis, inplane_axes, method = area_from_facet(fpath, slab)

        # per-model slab energies, gamma, deltaE/atom
        E_slab_models = []
        gamma_eV_A2_models = []
        gamma_J_m2_models = []
        deltaE_per_atom_models = []

        for k, mp in enumerate(model_paths):
            E_slab = get_mace_energy(slab.copy(), mp, device=device, head="target_head")
            E_slab_models.append(E_slab)
            # gamma
            gamma_eV_A2 = (E_slab - n_fu * E_bulk_per_fu_models[k]) / (2.0 * area_A2)
            gamma_J_m2  = gamma_eV_A2 * (EV_TO_J / A2_TO_M2)
            gamma_eV_A2_models.append(gamma_eV_A2)
            gamma_J_m2_models.append(gamma_J_m2)
            # deltaE per atom (requested)
            deltaE_pa = (E_slab - E_bulk_models[k]) / n_atoms
            deltaE_per_atom_models.append(deltaE_pa)

        # ensemble averages
        E_slab_avg = float(np.mean(E_slab_models))
        gamma_eV_A2_avg = float(np.mean(gamma_eV_A2_models))
        gamma_J_m2_avg  = float(np.mean(gamma_J_m2_models))
        deltaE_per_atom_avg = float(np.mean(deltaE_per_atom_models))

        facet, term = parse_facet_and_term(fpath)

        # Detailed per-model row
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
            # Ensemble-averaged (headline)
            "gamma_eV_A2_avg": gamma_eV_A2_avg,
            "gamma_J_m2_avg": gamma_J_m2_avg,
            # Requested ΔE per atom (name includes ensemble)
            f"deltaE_eV_per_atom_{ensemble_name}": deltaE_per_atom_avg,
        }

        # Attach per-model columns
        for k, mp in enumerate(model_paths, start=1):
            tag = f"m{k}"
            row[f"E_bulk_eV_{tag}"] = E_bulk_models[k-1]
            row[f"E_bulk_per_fu_eV_{tag}"] = E_bulk_per_fu_models[k-1]
            row[f"E_bulk_per_atom_eV_{tag}"] = E_bulk_per_atom_models[k-1]
            row[f"E_slab_eV_{tag}"] = E_slab_models[k-1]
            row[f"gamma_eV_A2_{tag}"] = gamma_eV_A2_models[k-1]
            row[f"gamma_J_m2_{tag}"] = gamma_J_m2_models[k-1]
            row[f"deltaE_eV_per_atom_{tag}"] = deltaE_per_atom_models[k-1]
            row[f"model_path_{tag}"] = mp

        rows_detail.append(row)

    df_detail = pd.DataFrame(rows_detail).sort_values(
        ["ensemble", "facet", "termination", "gamma_J_m2_avg", "file"],
        na_position="last"
    ).reset_index(drop=True)

    # Summary: group by slab file (ensemble already single-valued here)
    keep_cols = [
        "ensemble", "file", "facet", "termination", "n_formula_units", "n_atoms",
        "area_A2", "gamma_eV_A2_avg", "gamma_J_m2_avg",
        f"deltaE_eV_per_atom_{ensemble_name}"
    ]
    df_summary = df_detail[keep_cols].copy()

    # Write outputs
    detail_xlsx = f"{out_prefix}_{ensemble_name}_detailed.xlsx"
    detail_json = f"{out_prefix}_{ensemble_name}_detailed.json"
    summary_xlsx = f"{out_prefix}_{ensemble_name}_summary.xlsx"
    summary_json = f"{out_prefix}_{ensemble_name}_summary.json"

    df_detail.to_excel(detail_xlsx, index=False)
    with open(detail_json, "w") as f:
        json.dump(df_detail.to_dict(orient="records"), f, indent=2)

    df_summary.to_excel(summary_xlsx, index=False)
    with open(summary_json, "w") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=2)

    print(f"[{ensemble_name}] wrote:")
    print("  ", detail_xlsx)
    print("  ", detail_json)
    print("  ", summary_xlsx)
    print("  ", summary_json)

def main():
    parser = argparse.ArgumentParser(description="Compute stoichiometric surface energies with MACE ensembles.")
    parser.add_argument("--base", default="/home/phanim/harshitrawat/summer/MS_LLZO_surface_data-master")
    parser.add_argument("--bulk", default=None, help="Path to bulk CIF; defaults to refs/LLZO_tet_bulk.cif under --base")
    parser.add_argument("--stoich_dir", default=None, help="Path to stoichiometric slabs; defaults to stoichiometric/ under --base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_prefix", default="surface_energy_mace")
    args = parser.parse_args()

    BASE = args.base
    BULK = args.bulk or os.path.join(BASE, "refs", "LLZO_tet_bulk.cif")
    STOICH = args.stoich_dir or os.path.join(BASE, "stoichiometric")

    # Ensembles you gave:
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

    # Run both ensembles
    eval_ensemble(
        base_dir=BASE,
        bulk_path=BULK,
        stoich_dir=STOICH,
        model_paths=MACE_T2,
        device=args.device,
        ensemble_name="MACET2",
        out_prefix=args.out_prefix
    )
    eval_ensemble(
        base_dir=BASE,
        bulk_path=BULK,
        stoich_dir=STOICH,
        model_paths=MACE_T1,
        device=args.device,
        ensemble_name="MACET1",
        out_prefix=args.out_prefix
    )

if __name__ == "__main__":
    main()
