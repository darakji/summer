from pymatgen.core.surface import SlabGenerator
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter
import os

API_KEY = "Yoa1b2uiwwxd5fpoSFS9aaTg7qSuvnF1"  # <-- Replace with your Materials Project API key
material_id = "mp-135"  # Li metal (BCC)

miller = (1, 1, 1)         # Choose (100), (110), or (111)
num_layers = 6             # Li atomic layers
vacuum = 10.0              # Å vacuum
center = True              # Center slab around z = 0

with MPRester(API_KEY) as mpr:
    li_bulk = mpr.get_structure_by_material_id(material_id)

slabgen = SlabGenerator(
    initial_structure=li_bulk,
    miller_index=miller,
    min_slab_size=num_layers,
    min_vacuum_size=vacuum,
    center_slab=center
)

slab = slabgen.get_slab()

# Save to CIF
os.makedirs("li_slabs", exist_ok=True)
CifWriter(slab).write_file("li_slabs/Li_111_slab.cif")

print("✓ Saved: li_slabs/Li_111_slab.cif")