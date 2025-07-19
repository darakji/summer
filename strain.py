import os
import pandas as pd
import numpy as np
from ase.io import read
from ase.calculators.calculator import CalculatorError
from chgnet.model.dynamics import CHGNetCalculator

# === Config ===
folders = {
    "/home/phanim/harshitrawat/mehul/summer/md/mdcifs": "/home/phanim/harshitrawat/mehul/summer/md/mdinfo_chgnet_predictions.xlsx",
    "/home/phanim/harshitrawat/mehul/summer/md/mdcifs_strained_perturbed": "/home/phanim/harshitrawat/mehul/summer/md/strain_perturb_chgnet_predictions.xlsx"
}

calc = CHGNetCalculator(device="cuda")  # or "cpu"
import torch
torch_device = "cuda"

# Now force the model device
calc.model.to(torch_device)
print("Now model is on:", next(calc.model.parameters()).device)

def extract_info_from_cif(cif_path):
    try:
        atoms = read(cif_path)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax = np.max(np.linalg.norm(forces, axis=1))
        stress = atoms.get_stress(voigt=False).tolist()  # 3x3 stress tensor
        magmom = atoms.get_magnetic_moment() if "magmom" in atoms.arrays else None

        return {
            "file": os.path.basename(cif_path),
            "energy_eV": energy,
            "fmax_eV_per_A": fmax,
            "stress_tensor": stress,
            "magmom_total": magmom
        }
    except (CalculatorError, Exception) as e:
        return {"file": os.path.basename(cif_path), "error": str(e)}

# === Run on both folders ===
for folder, output_excel in folders.items():
    print(f"\n📂 Processing folder: {folder}")
    results = []

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".cif"):
            path = os.path.join(folder, fname)
            result = extract_info_from_cif(path)
            results.append(result)
            print(f"✅ {fname}" if "error" not in result else f"❌ {fname} — {result['error']}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n🧾 Saved predictions to: {output_excel}")
