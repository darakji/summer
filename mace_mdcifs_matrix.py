import os
import pandas as pd
import numpy as np
from ase.io import read
from ase.calculators.calculator import CalculatorError
from mace.calculators import MACECalculator, load_model

# === Config ===
folder = "/home/mehuldarak/summer/md/mdcifs_strained_perturbed"
output_excel = "/home/mehuldarak/summer/md/mdinfo_mace_predictions_perturb.xlsx"
model_path = "/home/mehuldarak/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model"

# === Load custom MACE model on GPU 0 ===
calc = MACECalculator(model=load_model(model_path), device="cuda:0")
print("✅ Loaded MACE model on cuda:0")

def extract_info_from_cif(cif_path):
    try:
        atoms = read(cif_path)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        fmax = np.max(np.linalg.norm(forces, axis=1))
        stress = atoms.get_stress(voigt=False).tolist()
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

results = []
for fname in sorted(os.listdir(folder)):
    if fname.endswith(".cif"):
        path = os.path.join(folder, fname)
        result = extract_info_from_cif(path)
        results.append(result)
        print(f"✅ {fname}" if "error" not in result else f"❌ {fname} — {result['error']}")

pd.DataFrame(results).to_excel(output_excel, index=False)
print(f"\n🧾 Saved predictions to: {output_excel}")
