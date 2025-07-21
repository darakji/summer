import os
import json
import pandas as pd
import numpy as np
from ase.io import read
from ase.calculators.calculator import CalculatorError
from chgnet.model.dynamics import CHGNetCalculator
import torch

# === User Config ===
cif_path = "/path/to/your/testfile.cif"  # 🔁 Replace with actual CIF path
output_excel = "chgnet_single_test_output.xlsx"

# === Setup Calculator ===
calc = CHGNetCalculator(device="cuda")  # or "cpu"
calc.model.to("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 CHGNet model loaded on:", next(calc.model.parameters()).device)

# === Extraction Function ===
def extract_info_from_cif(path):
    try:
        atoms = read(path)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False).tolist()
        magmom = atoms.get_magnetic_moment() if "magmom" in atoms.arrays else None

        return {
            "file": os.path.basename(path),
            "energy_eV": energy,
            "forces_per_atom_eV_per_A": json.dumps(forces.tolist()),
            "stress_tensor": json.dumps(stress),
            "magmom_total": magmom,
            "natoms": len(atoms),
        }
    except (CalculatorError, Exception) as e:
        return {"file": os.path.basename(path), "error": str(e)}

# === Run ===
result = extract_info_from_cif(cif_path)
df = pd.DataFrame([result])
df.to_excel(output_excel, index=False)

print("\n✅ Done. Results written to:", output_excel)
