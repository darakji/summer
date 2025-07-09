from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import os

# === CONFIGURATION ===
lco_dir = "LCO_TaskerTypeI_slabs"               # Folder with LCO slabs
llzo_dir = "llzo_lco_slabs"                     # Folder with LLZO slabs
output_dir = "LCO_LLZO_interfaces_fixed"        # Where to save interfaces
os.makedirs(output_dir, exist_ok=True)

target_a, target_b = 12.70, 13.20  # in-plane match with LLZO
gap = 4.0                          # Å between slabs
vacuum = 15.0                      # Å above and below
lco_z_repeats = 2                 # increase for thicker LCO
lco_z_gap = 2.0                   # Å gap between LCO repeats

def compute_repetition(cell, target_a, target_b):
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    return int(np.round(target_a / a_len)), int(np.round(target_b / b_len))

def build_thick_lco_unit(slab, z_repeats=2, z_gap=2.0):
    """Stack LCO slab multiple times in z with spacing."""
    slabs = []
    slab_thickness = np.max(slab.positions[:, 2]) - np.min(slab.positions[:, 2])
    for i in range(z_repeats):
        slab_i = slab.copy()
        slab_i.positions[:, 2] += i * slab_thickness   # ← no extra gap
        slabs.append(slab_i)
    combined = slabs[0]
    for slab_i in slabs[1:]:
        combined += slab_i
    return combined

for lco_file in os.listdir(lco_dir):
    if not lco_file.endswith(".cif"):
        continue
    lco = read(os.path.join(lco_dir, lco_file))
    rep_a, rep_b = compute_repetition(lco.get_cell(), target_a, target_b)
    lco_super = make_supercell(lco, np.diag([rep_a, rep_b, 1]))
    lco_super.positions[:, 2] -= lco_super.positions[:, 2].min()

    lco_thick = build_thick_lco_unit(lco_super, lco_z_repeats, lco_z_gap)
    lco_thick.positions[:, 2] -= np.min(lco_thick.positions[:, 2])
    lco_thickness = np.max(lco_thick.positions[:, 2])

    for llzo_file in os.listdir(llzo_dir):
        if not llzo_file.endswith(".cif"):
            continue
        llzo = read(os.path.join(llzo_dir, llzo_file))
        llzo.positions[:, 2] -= llzo.positions[:, 2].min()
        llzo.positions[:, 2] += vacuum
        llzo_top = np.max(llzo.positions[:, 2])  # this must come AFTER vacuum shift


        lco_thick.positions[:, 2] += llzo_top + gap + vacuum
        combined = llzo + lco_thick

        total_c = vacuum + llzo_top + gap + lco_thickness + vacuum
        cell = combined.get_cell()
        cell[2, 2] = total_c
        combined.set_cell(cell)
        combined.wrap()

        out_name = f"{lco_file[:-4]}__{llzo_file[:-4]}_interface.cif"
        write(os.path.join(output_dir, out_name), combined)
        print(f"✅ Saved: {out_name}")
