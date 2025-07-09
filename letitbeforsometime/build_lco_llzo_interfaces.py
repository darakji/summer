from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import os

# === CONFIGURATION ===
lco_dir = "LCO_TaskerTypeI_slabs"
llzo_dir = "llzo_lco_slabs"
output_dir = "LCO_LLZO_interfaces"
os.makedirs(output_dir, exist_ok=True)

target_a, target_b = 12.70, 13.20  # LLZO in-plane dimensions
gap = 4.0                          # Å gap between LLZO and LCO
vacuum = 15.0                      # Å vacuum above and below

def compute_repetition(cell, target_a, target_b):
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    rep_a = int(np.round(target_a / a_len))
    rep_b = int(np.round(target_b / b_len))
    return rep_a, rep_b

for lco_file in os.listdir(lco_dir):
    if not lco_file.endswith(".cif"):
        continue

    # === Load and expand LCO slab ===
    lco = read(os.path.join(lco_dir, lco_file))
    rep_a, rep_b = compute_repetition(lco.get_cell(), target_a, target_b)
    lco_super = make_supercell(lco, np.diag([rep_a, rep_b, 1]))
    lco_super.positions[:, 2] -= lco_super.positions[:, 2].min()
    lco_thick = np.max(lco_super.positions[:, 2])

    for llzo_file in os.listdir(llzo_dir):
        if not llzo_file.endswith(".cif"):
            continue

        # === Load LLZO slab ===
        llzo = read(os.path.join(llzo_dir, llzo_file))
        llzo.positions[:, 2] -= llzo.positions[:, 2].min()
        llzo.positions[:, 2] += vacuum
        llzo_thick = np.max(llzo.positions[:, 2]) - np.min(llzo.positions[:, 2])

        # === Position LCO slab above LLZO + gap ===
        lco_super.positions[:, 2] += vacuum + llzo_thick + gap

        # === Combine structures ===
        combined = llzo + lco_super

        # === Set total z-length to exactly 15 + t_LLZO + 4 + t_LCO + 15 ===
        total_c = vacuum + llzo_thick + gap + lco_thick + vacuum
        cell = combined.get_cell()
        cell[2, 2] = total_c
        combined.set_cell(cell)
        combined.wrap()

        # === Save output ===
        out_name = f"{lco_file[:-4]}__{llzo_file[:-4]}_interface.cif"
        write(os.path.join(output_dir, out_name), combined)
        print(f"✅ Saved: {out_name}")
