# Re-run due to state reset
import os
import random
import shutil
import pandas as pd
from ase import io
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

from chgnet.model.dynamics import CHGNetCalculator

# === CONFIGURATION ===
relaxed_root = "relax_final"
md_root = "md"
traj_dir = os.path.join(md_root, "mdtraj_prime")
cif_dir = os.path.join(md_root, "mdcifs_prime")
picked_dir = os.path.join(md_root, "picked400_cifs")

temperatures = [350, 420]           # Run MD at these temperatures
timestep = 1 * units.fs             # MD timestep: 1 femtosecond
n_steps = 1000                      # Run for 1000 steps (1 ps)
snapshot_interval = 5              # Save a snapshot every 5 steps
friction = 0.01                    # Langevin damping

# Ensure output dirs exist
os.makedirs(traj_dir, exist_ok=True)
os.makedirs(cif_dir, exist_ok=True)
os.makedirs(picked_dir, exist_ok=True)

# === UTILITY ===
def parse_filename_metadata(fname: str) -> dict:
    tokens = os.path.splitext(fname)[0].split("_")
    try:
        return dict(
            full_name      = fname,
            llzo_dir       = tokens[2],
            llzo_termination = tokens[3],
            llzo_order     = tokens[4],
            stoichiometry  = tokens[5],
            li_dir         = tokens[7],
            notes          = ""
        )
    except Exception as e:
        return dict(full_name=fname, llzo_dir=None, llzo_termination=None,
                    llzo_order=None, stoichiometry=None, li_dir=None,
                    notes=f"parse_error:{e}")

# === MD JOB FUNCTION ===
def run_md_task(cif_path: str, base_name: str, T: int) -> dict:
    atoms = io.read(cif_path)
    atoms.calc = CHGNetCalculator()
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    dyn = Langevin(atoms, timestep, temperature_K=T, friction=friction)

    traj_path = os.path.join(traj_dir, f"{base_name}_T{T}.traj")
    traj_writer = io.Trajectory(traj_path, 'w', atoms)
    snapshots = []

    def save_snapshot(a=atoms):
        traj_writer.write(a)
        snapshots.append(a.copy())

    dyn.attach(save_snapshot, interval=snapshot_interval)
    dyn.run(n_steps)
    traj_writer.close()

    for i, atoms_snap in enumerate(snapshots):
        cif_file = f"{base_name}_T{T}_{i:04d}.cif"
        cif_path_out = os.path.join(cif_dir, cif_file)
        io.write(cif_path_out, atoms_snap)

    return dict(
        base_name=base_name,
        temperature_K=T,
        n_snapshots=len(snapshots),
        source_path=cif_path
    )

# === JOB LIST GENERATION ===
jobs = []
for root, _, files in os.walk(relaxed_root):
    for fname in files:
        if fname.endswith(".cif") and fname.startswith("cellrelaxed_"):
            cif_path = os.path.join(root, fname)
            base_name = os.path.splitext(fname)[0]
            base_meta = parse_filename_metadata(fname)
            for T in temperatures:
                jobs.append((cif_path, base_name, T, base_meta))

# === PARALLEL EXECUTION ===
def pool_wrapper(argtuple):
    cif_path, base_name, T, base_meta = argtuple
    result = run_md_task(cif_path, base_name, T)
    result.update(base_meta)
    return result

try:
    set_start_method("spawn", force=False)
except RuntimeError:
    pass

with Pool(processes=2) as pool:
    results = list(tqdm(pool.imap_unordered(pool_wrapper, jobs),
                        total=len(jobs),
                        desc="MD progress"))

# === SAVE METADATA ===
df = pd.DataFrame(results)
excel_path = os.path.join(md_root, "mdinfo.xlsx")
df.to_excel(excel_path, index=False)

# === RANDOM 400 PICK ===
all_cifs = []
for root, _, files in os.walk(cif_dir):
    for f in files:
        if f.endswith(".cif") and "_T" in f:
            all_cifs.append(os.path.join(root, f))

n_pick = min(400, len(all_cifs))
picked_cifs = random.sample(all_cifs, n_pick)

for cif_path in picked_cifs:
    shutil.copy2(cif_path, picked_dir)

# Output for user
import ace_tools as tools; tools.display_dataframe_to_user(name="MD Simulation Metadata", dataframe=df)
