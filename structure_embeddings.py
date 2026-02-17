import os
import json
from ase.io import read

BASE_DIR = "/home/phanim/harshitrawat/summer/universal_embeddings_results"
OUT_JSON = os.path.join(BASE_DIR, "structure_level_latents.json")

results = []

for fname in sorted(os.listdir(BASE_DIR)):
    if not fname.endswith(".xyz"):
        continue

    xyz_path = os.path.join(BASE_DIR, fname)

    # read ALL structures in the xyz file
    atoms_list = read(xyz_path, index=":")

    for atoms in atoms_list:
        struct_file = atoms.info["file"]          # original structure filename
        energy = atoms.info["mace_energy"]         # structure energy

        latents = atoms.arrays["mace_latent"]      # (N_atoms, D)
        struct_emb = latents.mean(axis=0)          # (D,)

        results.append(
            {
                "file": struct_file,
                "structure_embedding": struct_emb.tolist(),
                "mace_energy": energy,
            }
        )

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} structures to {OUT_JSON}")
