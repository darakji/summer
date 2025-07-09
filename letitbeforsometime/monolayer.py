import os
import numpy as np
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core import Structure

def build_true_li_half_monolayer_slab(structure: Structure, li_layer_thickness=2.5, vacuum=15.0):
    slab = structure.copy()
    coords = np.array([site.coords for site in slab])
    z_coords = coords[:, 2]
    z_max, z_min = z_coords.max(), z_coords.min()

    # Identify Li atoms near top/bottom
    li_sites = [(i, site) for i, site in enumerate(slab) if site.species_string == "Li"]
    top_li = sorted([s for s in li_sites if s[1].coords[2] > z_max - li_layer_thickness], key=lambda x: -x[1].coords[2])
    bot_li = sorted([s for s in li_sites if s[1].coords[2] < z_min + li_layer_thickness], key=lambda x: x[1].coords[2])

    # Keep 50% of Li atoms from top/bottom
    top_keep = [i for j, (i, _) in enumerate(top_li) if j % 2 == 0]
    bot_keep = [i for j, (i, _) in enumerate(bot_li) if j % 2 == 0]
    keep_indices = set(top_keep + bot_keep)

    # Keep middle slab and ½ Li on top/bottom
    new_sites = []
    for i, site in enumerate(slab):
        z = site.coords[2]
        if site.species_string == "Li":
            if i in keep_indices:
                new_sites.append(site)
        elif (z_min + li_layer_thickness < z < z_max - li_layer_thickness):
            new_sites.append(site)

    # Build and center structure
    new_slab = Structure.from_sites(new_sites)
    # Compute center of mass along z
    masses = np.array([site.specie.atomic_mass for site in new_slab])
    coords = np.array([site.coords for site in new_slab])
    center_of_mass_z = np.average(coords[:, 2], weights=masses)
    new_slab.translate_sites(range(len(new_slab)), [0, 0, -center_of_mass_z])

    # Add vacuum
    lattice = np.array(new_slab.lattice.matrix, copy=True)
    lattice[2][2] += vacuum
    # Rebuild structure with new lattice and same fractional coordinates
    new_slab = Structure(lattice, new_slab.species, new_slab.frac_coords, coords_are_cartesian=False)

    return new_slab

# === USAGE ===
input_dir = "lco_slabs"  # replace with your slab folder
output_dir = "lco_slabs_true_half_ml"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".cif"):
        struct = CifParser(os.path.join(input_dir, fname)).parse_structures()[0]
        new = build_true_li_half_monolayer_slab(struct)
        CifWriter(new).write_file(f"{output_dir}/{fname.replace('.cif', '_Li_half_ML_true.cif')}")
        print("✓", fname)