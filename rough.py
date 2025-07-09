from ase.io import read
import os
import numpy as np

folder = "llzo_lco_slabs"  # your extracted directory
for fname in os.listdir(folder):
    if fname.endswith(".cif"):
        atoms = read(os.path.join(folder, fname))
        cell = atoms.get_cell()
        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        print(f"{fname}: a = {a:.2f} Å, b = {b:.2f} Å")
