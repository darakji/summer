from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.cif import CifWriter
import os

# Initialize MP API
mpr = MPRester("Yoa1b2uiwwxd5fpoSFS9aaTg7qSuvnF1")  # Replace with your Materials Project API key

# Get R-3m LiCoO2 structure
entries = mpr.summary.search(formula="LiCoO2", fields=["structure", "material_id"])
structure = entries[0].structure  # R-3m phase

# Define slab parameters
surfaces = {"001": [0, 0, 1], "104": [1, 0, 4], "110": [1, 1, 0]}
min_slab_thickness = 10  # in Å
min_vacuum_thickness = 15  # in Å

# Output directory
os.makedirs("lco_slabs", exist_ok=True)

# Generate slabs
for label, miller in surfaces.items():
    gen = SlabGenerator(
        initial_structure=structure,
        miller_index=miller,
        min_slab_size=min_slab_thickness,
        min_vacuum_size=min_vacuum_thickness,
        center_slab=True,
        in_unit_planes=True,
    )
    slabs = gen.get_slabs(symmetrize=True)
    for i, slab in enumerate(slabs[:2]):  # up to 2 terminations per facet
        fname = f"lco_slabs/LCO_{label}_term{i}.cif"
        CifWriter(slab).write_file(fname)
        print(f"Saved: {fname}")
