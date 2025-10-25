#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from ase.io import read
from ase.calculators.calculator import CalculatorError
from mace.calculators import MACECalculator

# === Config ===
base_dir = "/home/phanim/harshitrawat/summer/md"
output_excel = os.path.join(base_dir, "MACE_T2_w4_predictions_onT1.xlsx")
model_path = "/home/phanim/harshitrawat/summer/mace_T2_w4_2heads_final_compiled.model"

# === Load MACE model
calc = MACECalculator(model_path=model_path, device="cuda:0", head = "target_head")
print("✅ Loaded MACE model on cuda:0")

def extract_info_from_cif(cif_path, folder):
    try:
        atoms = read(cif_path)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax = np.max(np.linalg.norm(forces, axis=1))
        stress = atoms.get_stress(voigt=False).tolist()

        return {
            "file": os.path.basename(cif_path),
            "subfolder": folder,
            "energy_eV": energy,
            "forces_per_atom_eV_per_A": forces,
            "stress_tensor": stress,
        }
    except (CalculatorError, Exception) as e:
        return {
            "file": os.path.basename(cif_path),
            "subfolder": folder,
            "error": str(e),
        }

results = []

# === Gather all CIF files
all_cif_files = []
for folder in ["mdcifs", "mdcifs_strained_perturbed"]:
    full_path = os.path.join(base_dir, folder)
    all_cif_files += [
        (os.path.join(full_path, fname), folder)
        for fname in sorted(os.listdir(full_path))
        if fname.endswith(".cif")
    ]

# === Run prediction with progress printing
total = len(all_cif_files)
for i, (cif_path, folder) in enumerate(all_cif_files, 1):
    fname = os.path.basename(cif_path)
    if "_perturbed.cif" in fname:
        folder = "mdcifs_strained_perturbed"
    else:
        folder = "mdcifs"

    result = extract_info_from_cif(cif_path, folder)
    results.append(result)

    progress_str = f"[{i}/{total}]"
    if "error" not in result:
        print(f"✅ {progress_str} {fname}")
    else:
        print(f"❌ {progress_str} {fname} — {result['error']}")

# === Save to Excel
pd.DataFrame(results).to_excel(output_excel, index=False)
print(f"\n🧾 Saved predictions to: {output_excel}")

# === Save to JSON (compact + full precision)
output_json = os.path.join(base_dir, "MACE_T2_w4_predictions_onT1.json")
import json
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)   # indent=2 for readable formatting

print(f"📁 Saved JSON to: {output_json}")
