from ase.io import read
import pandas as pd
import json

# Load Excel
df = pd.read_excel("/home/phanim/harshitrawat/summer/T1_chgnet_labeled.xlsx")
group = df[df["file"] == "cellrelaxed_LLZO_001_Zr_code93_sto__Li_100_slab_heavy_T300_0002.cif"]

# Load CIF
atoms = read("/home/phanim/harshitrawat/summer/md/mdcifs/cellrelaxed_LLZO_001_Zr_code93_sto__Li_100_slab_heavy_T300_0002.cif")

# Inject energy, forces, stress
atoms.info["energy"] = float(group["energy_eV"].iloc[0])
atoms.arrays["forces"] = json.loads(group["forces_per_atom_eV_per_A"].iloc[0])
atoms.info["stress"] = json.loads(group["stress_tensor"].iloc[0])
