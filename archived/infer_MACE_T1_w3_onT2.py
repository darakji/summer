#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from ase.io import iread
from ase.calculators.calculator import CalculatorError
from mace.calculators import MACECalculator

# === Config ===
BASE_DIR = "/home/phanim/harshitrawat/summer"
DATA_DIR = os.path.join(BASE_DIR, "T1_T2_T3_data")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints/mace_T1_w3_2heads_final_run-40.model")

DEVICE = "cuda:0"
HEAD = "target_head"

OUTPUT_DIR = os.path.join(BASE_DIR, "md")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "MACE_T1_w3_predictions_onT2T3.json")
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "MACE_T1_w3_predictions_onT2T3.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load model ===
calc = MACECalculator(model_path=MODEL_PATH, device=DEVICE, head=HEAD)
print(f"✅ Loaded MACE_T1 model on {DEVICE} (head={HEAD})")

# === Helper ===
def extract_info_from_frame(atoms, dataset_tag):
    """Compute energy, forces, and stress for one frame."""
    try:
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float).tolist()
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float).tolist()
        return {
            "dataset": dataset_tag,
            "file": atoms.info.get("file", None),
            "energy_eV": energy,
            "forces_per_atom_eV_per_A": forces,
            "stress_tensor": stress,
            "error": None,
        }
    except (CalculatorError, Exception) as e:
        return {
            "dataset": dataset_tag,
            "file": atoms.info.get("file", None),
            "energy_eV": None,
            "forces_per_atom_eV_per_A": None,
            "stress_tensor": None,
            "error": str(e),
        }

def run_predictions(extxyz_path, tag):
    """Run predictions for all frames in an extxyz file."""
    results = []
    frames = list(iread(extxyz_path, index=":"))
    total = len(frames)
    print(f"\n▶️ Running predictions on {total} frames from {extxyz_path}")

    for i, atoms in enumerate(frames, 1):
        result = extract_info_from_frame(atoms, tag)
        results.append(result)
        fname = result.get("file", f"{tag}_frame_{i}")
        progress = f"[{i}/{total}]"
        if result["error"] is None:
            print(f"✅ {progress} {fname}")
        else:
            print(f"❌ {progress} {fname} — {result['error']}")
    return results

# === Run on T2 and T3 ===
t2_path = os.path.join(DATA_DIR, "T2_chgnet_labeled.extxyz")
t3_path = os.path.join(DATA_DIR, "T3_chgnet_labeled.extxyz")

combined_results = []
if os.path.isfile(t2_path):
    combined_results.extend(run_predictions(t2_path, "T2"))
else:
    print(f"⚠️ T2 file not found: {t2_path}")

if os.path.isfile(t3_path):
    combined_results.extend(run_predictions(t3_path, "T3"))
else:
    print(f"⚠️ T3 file not found: {t3_path}")

# === Save combined outputs ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(combined_results, f, indent=2)
print(f"\n📁 Saved combined JSON: {OUTPUT_JSON}")

pd.DataFrame(combined_results).to_excel(OUTPUT_EXCEL, index=False)
print(f"🧾 Saved combined Excel: {OUTPUT_EXCEL}")

print("\n✅ All done! Combined T2+T3 results written successfully.")
