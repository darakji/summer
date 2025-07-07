import os
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

input_dir = "summer_llzo_cifs"
output_dir = "summer_llzo_cifs_prepared"
os.makedirs(output_dir, exist_ok=True)

MIN_SLAB_THICKNESS = 10.0  # Å
VACUUM_PADDING = 30.0      # Å
OVERLAP_THRESHOLD = 1.4    # Å

log = []

def compute_z_metrics(struct):
    z = [site.coords[2] for site in struct]
    return min(z), max(z), max(z) - min(z)

def compute_min_distance(struct):
    d = struct.distance_matrix
    np.fill_diagonal(d, np.inf)
    return np.min(d)

for fname in os.listdir(input_dir):
    if fname.endswith(".cif"):
        try:
            struct = Structure.from_file(os.path.join(input_dir, fname))
            zmin, zmax, thick = compute_z_metrics(struct)
            mindist = compute_min_distance(struct)

            if thick < MIN_SLAB_THICKNESS:
                log.append((fname, round(thick, 2), round(mindist, 2), None, "Skipped: too thin"))
                continue
            if mindist < OVERLAP_THRESHOLD:
                log.append((fname, round(thick, 2), round(mindist, 2), None, "Skipped: atom overlap"))
                continue

            c_final = thick + VACUUM_PADDING
            struct.translate_sites(range(len(struct)), [0, 0, 0.5 * c_final - 0.5 * (zmax + zmin)])

            new_lattice = Lattice.from_parameters(
                a=struct.lattice.a, b=struct.lattice.b, c=c_final,
                alpha=struct.lattice.alpha, beta=struct.lattice.beta, gamma=struct.lattice.gamma
            )

            new_struct = Structure(new_lattice, struct.species, [s.coords for s in struct], coords_are_cartesian=True)
            new_struct.to(filename=os.path.join(output_dir, fname))
            log.append((fname, round(thick, 2), round(mindist, 2), round(c_final, 2), "Success"))

        except Exception as e:
            log.append((fname, None, None, None, f"Failed: {e}"))

df = pd.DataFrame(log, columns=["File", "Slab Thickness (Å)", "Min Distance (Å)", "Final c (Å)", "Status"])
df.to_csv("slab_prep_log.csv", index=False)
print(df)