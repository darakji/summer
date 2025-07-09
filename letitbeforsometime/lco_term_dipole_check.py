import os
import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser

def analyze_slab_properties(structure, surface_thickness=2.5):
    coords = np.array([site.coords for site in structure])
    z_coords = coords[:, 2]
    z_max, z_min = z_coords.max(), z_coords.min()

    top_sites = [site for site in structure if site.coords[2] > z_max - surface_thickness]
    bottom_sites = [site for site in structure if site.coords[2] < z_min + surface_thickness]

    def species_set(sites):
        return set(site.species_string for site in sites)

    top_species = species_set(top_sites)
    bottom_species = species_set(bottom_sites)
    symmetric = top_species == bottom_species

    # Approximate dipole along z using electronegativity-weighted z
    en = {'Li': 1.0, 'O': 3.5, 'Co': 1.9}
    weighted_z = sum(en.get(site.species_string, 2.0) * site.coords[2] for site in structure)
    total_en = sum(en.get(site.species_string, 2.0) for site in structure)
    dipole_z = weighted_z / total_en if total_en else 0.0

    # Heuristic charge neutrality
    elems = [site.species_string for site in structure]
    charge_neutral = abs(elems.count('Li') - elems.count('O') - elems.count('Co')) < 5

    return {
        "estimated_dipole_z": round(dipole_z, 2),
        "symmetric_surfaces": symmetric,
        "top_species": ','.join(sorted(top_species)),
        "bottom_species": ','.join(sorted(bottom_species)),
        "charge_neutral": charge_neutral
    }

# === Run ===
input_dir = "lco_slabs"  # replace with your slab folder
results = []

for fname in os.listdir(input_dir):
    if fname.endswith(".cif"):
        struct = CifParser(os.path.join(input_dir, fname)).get_structures()[0]
        props = analyze_slab_properties(struct)
        props["filename"] = fname
        results.append(props)

df = pd.DataFrame(results)
df.to_excel("lco_slab_analysis.xlsx", index=False)
print(df)
