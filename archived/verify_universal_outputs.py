
import ase.io
import sys
import os

FILES = [
    "universal_embeddings_results/Universal_on_T1.xyz",
    "universal_embeddings_results/Universal_on_T2.xyz",
    "universal_embeddings_results/Universal_on_T3.xyz"
]

def check_file(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print(f"  [MISSING] File not found.")
        return

    try:
        # Read first frame
        atoms = ase.io.read(path, index=0)
        
        # Check Energy
        if "mace_energy" in atoms.info:
            e = atoms.info["mace_energy"]
            print(f"  [OK] Energy found: {e:.4f} eV")
        else:
            print(f"  [FAIL] 'mace_energy' missing from atoms.info")

        # Check Latents
        if "mace_latent" in atoms.arrays:
            lat = atoms.arrays["mace_latent"]
            print(f"  [OK] Latents found. Shape: {lat.shape}")
            if lat.shape[1] == 128:
                 print(f"  [OK] Latent dimension is 128.")
            else:
                 print(f"  [WARN] Latent dimension is {lat.shape[1]}, expected 128.")
        else:
            print(f"  [FAIL] 'mace_latent' missing from atoms.arrays")

    except Exception as e:
        print(f"  [ERROR] Could not read file: {e}")

if __name__ == "__main__":
    print("-" * 40)
    for f in FILES:
        check_file(f)
        print("-" * 40)
