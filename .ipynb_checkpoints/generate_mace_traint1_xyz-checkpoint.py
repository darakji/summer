import torch
# permanently allow `slice` in the safe unpickler
torch.serialization.add_safe_globals([slice])

import os
import json
import pandas as pd
import numpy as np
from ase.io import read
from ase.db import connect
from chgnet.model.dynamics import CHGNetCalculator
from ase.calculators.calculator import CalculatorError
import torch
print("📥 Starting... Loading Excel and preparing to write DB/XYZ")

# === Paths ===
orig_excels = [
    "/home/phanim/harshitrawat/summer/md/mdinfo_chgnet_predictions_forces.xlsx",
    "/home/phanim/harshitrawat/summer/md/strain_perturb_chgnet_predictions_forces.xlsx"
]
t1_split = "/home/phanim/harshitrawat/summer/md/T1_chgnet_labeled.xlsx"
base_dir   = "/home/phanim/harshitrawat/summer/md"
db_path    = "mace_train_T1.db"

# === Load & filter ===
df_orig = pd.concat([pd.read_excel(x) for x in orig_excels], ignore_index=True)
t1_files = set(pd.read_excel(t1_split)["file"])
df = df_orig[df_orig["file"].isin(t1_files)].reset_index(drop=True)

print(f"📂 Total T1 entries: {len(df)}")

# === Prepare lists ===
good = []   # tuples of (fname, E, F, S)
bad  = []   # list of fnames needing CHGNet

for row in df.itertuples(index=False):
    fname = row.file
    try:
        E = float(row.energy_eV)
        F = json.loads(row.forces_per_atom_eV_per_A)
        S = json.loads(row.stress_tensor)
        # quick length check
        # we’ll verify atom count later
        good.append((fname, E, F, S))
    except Exception:
        bad.append(fname)

print(f"✅ Good JSON: {len(good)}, ⚠️ Bad JSON entries: {len(bad)}")

# === Helper ===
def resolve_cif(fname):
    sub = "mdcifs_strained_perturbed" if "perturbed" in fname else "mdcifs"
    return os.path.join(base_dir, sub, fname)

# === Open DB ===
n_ok = n_fail = n_recomputed = 0
with connect(db_path, append=False) as db:

    # 1) Write all good entries
    print("\n▶️ Writing all valid entries to DB…")
    for i, (fname, E, F, S) in enumerate(good, 1):
        path = resolve_cif(fname)
        if not os.path.exists(path):
            print(f"❌ [{i}/{len(good)}] Missing CIF: {fname}")
            n_fail += 1
            continue
        try:
            atoms = read(path)
            if len(F) != len(atoms):
                raise ValueError("Atom/force count mismatch")
            atoms.info["energy"]  = E
            atoms.info["stress"]  = S
            atoms.arrays["forces"] = F
            db.write(atoms)
            n_ok += 1
            print(f"✅ [{i}/{len(good)}] {fname}")
        except Exception as e:
            print(f"❌ [{i}/{len(good)}] {fname} — {e}")
            n_fail += 1

    # 2) Recompute bad entries on GPU
    if bad:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n▶️ Recomputing {len(bad)} bad entries on {device}…")
        calc = CHGNetCalculator(device=device)
        calc.model.to(device)

        for j, fname in enumerate(bad, 1):
            path = resolve_cif(fname)
            if not os.path.exists(path):
                print(f"❌ [{j}/{len(bad)}] Missing CIF: {fname}")
                n_fail += 1
                continue
            try:
                atoms = read(path)
                atoms.calc = calc
                E = atoms.get_potential_energy()
                F = atoms.get_forces().tolist()
                S = atoms.get_stress(voigt=False).tolist()
                if len(F) != len(atoms):
                    raise ValueError("Atom/force count mismatch")
                atoms.info["energy"]  = float(E)
                atoms.info["stress"]  = S
                atoms.arrays["forces"] = F
                db.write(atoms)
                n_recomputed += 1
                print(f"🛠 [{j}/{len(bad)}] Recomputed {fname}")
            except CalculatorError as e:
                print(f"⚠️ [{j}/{len(bad)}] CHGNet failed: {fname} — {e}")
                n_fail += 1
            except Exception as e:
                print(f"❌ [{j}/{len(bad)}] {fname} — {e}")
                n_fail += 1

print(f"\n🎉 Done — OK: {n_ok}, Recomputed: {n_recomputed}, Failed: {n_fail}")
print(f"📦 DB written to: {db_path}")
