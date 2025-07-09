from ase.io import read, write
from ase.build import surface
import os

# === Load the hexagonal LiCoO2 CIF file ===
lco_bulk = read("LiCoO2_hexagonal.cif")

# === Configuration ===
vacuum = 15.0
layers = 6
output_dir = "LCO_TaskerTypeI_slabs"
os.makedirs(output_dir, exist_ok=True)

# === Define Tasker Type I surfaces using hexagonal axes ===
surfaces = {
    "10-10": (1, 0, 0),
    "11-20": (1, 1, 0),
    "10-14": (1, 0, 4)
}

# === Generate and save slabs ===
for name, miller in surfaces.items():
    slab = surface(lco_bulk, miller, layers=layers, vacuum=vacuum)
    slab.center(axis=2, vacuum=vacuum)
    out_file = os.path.join(output_dir, f"LCO_{name}_slab.cif")
    write(out_file, slab)
    print(f"Saved: {out_file}")
