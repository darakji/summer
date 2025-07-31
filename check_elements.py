from ase.io import read
from collections import Counter

atoms = read("/home/phanim/harshitrawat/summer/md/mdcifs_strained_perturbed/cellrelaxed_LLZO_001_Zr_code93_sto__Li_100_slab_heavy_T300_0022_strain-3_perturbed.cif")
elements = atoms.get_chemical_symbols()
counts = Counter(elements)

print("Unique elements:", sorted(counts.keys()))
print("Counts:", counts)
