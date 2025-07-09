import os
import numpy as np
from pymatgen.io.cif import CifParser

def check_half_monolayer_li_termination(structure, surface_thickness=2.5, tolerance=0.5):
    z_coords = np.array([site.coords[2] for site in structure])
    z_max, z_min = z_coords.max(), z_coords.min()

    top_layer = [site for site in structure if site.coords[2] > z_max - surface_thickness]
    bottom_layer = [site for site in structure if site.coords[2] < z_min + surface_thickness]

    # 1. Check that all top/bottom atoms are Li
    top_ok = all(site.species_string == "Li" for site in top_layer)
    bottom_ok = all(site.species_string == "Li" for site in bottom_layer)

    # 2. Check symmetry: average z of top Li vs reflected bottom Li
    top_z = np.mean([site.coords[2] for site in top_layer]) if top_layer else 0
    bottom_z = np.mean([site.coords[2] for site in bottom_layer]) if bottom_layer else 0
    is_symmetric = abs((z_max - top_z) - (bottom_z - z_min)) < tolerance

    return top_ok, bottom_ok, is_symmetric, len(top_layer), len(bottom_layer)

# === USAGE ===
input_dir = "lco_slabs_true_half_ml"  # path to your output folder

for fname in os.listdir(input_dir):
    if fname.endswith(".cif"):
        structure = CifParser(os.path.join(input_dir, fname)).get_structures()[0]
        top_ok, bottom_ok, symmetric, top_count, bot_count = check_half_monolayer_li_termination(structure)

        print(f"🧪 {fname}")
        print(f"   Top layer Li only:      {'✅' if top_ok else '❌'}")
        print(f"   Bottom layer Li only:   {'✅' if bottom_ok else '❌'}")
        print(f"   Symmetric:              {'✅' if symmetric else '❌'}")
        print(f"   Top Li count:           {top_count}")
        print(f"   Bottom Li count:        {bot_count}")
        print("—" * 60)
