# -----------------------------------------------------------------------------
# true_li_half_monolayer.py
# -----------------------------------------------------------------------------
import os
import numpy as np
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core import Structure

def true_li_half_ml_slab(
        structure: Structure,
        coverage: float = 0.5,         # 0.5  → half-monolayer; use 1.0 for full ML
        li_layer_thickness: float = 2.0,  # Å window to collect surface-Li atoms
        vacuum: float = 15.0):            # Å vacuum to add on c–axis
    """
    Return a slab with Li-only terminations at the top & bottom, each
    containing `coverage` × 100 % of the Li atoms in that layer.
    """
    slab = structure.copy()
    z = np.array([site.coords[2] for site in slab])
    z_max, z_min = z.max(), z.min()

    # ----- 1. collect Li atoms belonging to the top & bottom Li planes -----
    li_indices_top = [i for i, s in enumerate(slab)
                      if s.species_string == "Li" and s.coords[2] > z_max - li_layer_thickness]
    li_indices_bot = [i for i, s in enumerate(slab)
                      if s.species_string == "Li" and s.coords[2] < z_min + li_layer_thickness]

    # ----- 2. choose a subset to keep  (coverage = 0.5 keeps half) ----------
    keep_top = li_indices_top[::int(round(1/coverage))]  # alt pattern
    keep_bot = li_indices_bot[::int(round(1/coverage))]

    keep_indices = set(keep_top + keep_bot)

    # ----- 3. build new site list  -----------------------------------------
    new_sites = []
    for i, site in enumerate(slab):
        zc = site.coords[2]

        # Delete everything ABOVE top-Li plane, except Li atoms we've chosen
        if zc > (z_max - li_layer_thickness):
            if site.species_string == "Li" and i in keep_indices:
                new_sites.append(site)
            # non-Li atoms => drop
        # Delete everything BELOW bottom-Li plane, except chosen Li
        elif zc < (z_min + li_layer_thickness):
            if site.species_string == "Li" and i in keep_indices:
                new_sites.append(site)
        # Keep all atoms in the middle region
        else:
            new_sites.append(site)

    new_slab = Structure.from_sites(new_sites)

    # ----- 4. recenter slab & add vacuum  -----------------------------------
    z = np.array([s.coords[2] for s in new_slab])
    thickness = z.max() - z.min()
    # shift COM to zero
    new_slab.translate_sites(range(len(new_slab)), [0, 0, - (z.max() + z.min()) / 2])
    # enlarge c-axis
    new_lattice = np.array(new_slab.lattice.matrix, copy=True)
    new_lattice[2, 2] = thickness + vacuum
    from pymatgen.core.lattice import Lattice
    lattice_obj = Lattice(new_lattice)
    new_slab = Structure(lattice_obj, new_slab.species, new_slab.frac_coords)
    return new_slab


# ----------------------------  BATCH DRIVER  --------------------------------
if __name__ == "__main__":
    input_dir  = "lco_slabs"                 #  <<<  folder with your original CIFs
    output_dir = "lco_slabs_half_ML_true"    #  <<<  results go here
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".cif"):
            struct = CifParser(os.path.join(input_dir, fname)).parse_structures()[0]
            fixed  = true_li_half_ml_slab(struct, coverage=0.5)
            out    = os.path.join(output_dir, fname.replace(".cif", "_Li_half_ML_true.cif"))
            CifWriter(fixed).write_file(out)
            print("✓", os.path.basename(out))