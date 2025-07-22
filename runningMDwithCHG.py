import os, sys, time
import pandas as pd
from multiprocessing import Pool, set_start_method
from functools import partial
from tqdm import tqdm                                    # progress-bar
from ase import io
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# ---- CHGNet ASE calculator ---------------------------
# (use the canonical import path for v0.4.x and v0.5.x)
from chgnet.model.dynamics import CHGNetCalculator
# ------------------------------------------------------

# === Config ===========================================
relaxed_root   = "relax_final"
md_root        = "md"
traj_dir       = os.path.join(md_root, "mdtraj")
cif_dir        = os.path.join(md_root, "mdcifs")
os.makedirs(traj_dir, exist_ok=True)
os.makedirs(cif_dir, exist_ok=True)

temperatures        = [300, 450]      # K
timestep            = 1 * units.fs    # 1 fs
n_steps             = 1000            # 1 ps
snapshot_interval   = 5               # save every 5 steps  → 200 snapshots
friction            = 0.01

# ======================================================
def parse_filename_metadata(fname: str) -> dict:
    """Parse metadata from
       cellrelaxed_LLZO_dir_term_order_stoLi_dir_slab_heavy.cif"""
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

# ------------------------------------------------------
def run_md_task(cif_path: str, base_name: str, T: int) -> dict:
    """Run one MD simulation and export trajectory + cif snapshots."""
    # Make sure output dirs exist inside worker (needed on some clusters)
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(cif_dir, exist_ok=True)

    atoms         = io.read(cif_path)
    atoms.calc    = CHGNetCalculator()

    # initialise velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn           = Langevin(atoms, timestep, temperature_K=T, friction=friction)

    traj_path     = os.path.join(traj_dir, f"{base_name}_T{T}.traj")
    traj_writer   = io.Trajectory(traj_path, 'w', atoms)

    snapshots = []

    def save_snapshot(a=atoms):
        traj_writer.write(a)
        snapshots.append(a.copy())

    dyn.attach(save_snapshot, interval=snapshot_interval)
    dyn.run(n_steps)
    traj_writer.close()

    # ---- export CIFs ---------------------------------
    for i, atoms_snap in enumerate(snapshots):
        cif_file     = f"{base_name}_T{T}_{i:04d}.cif"
        cif_path_out = os.path.join(cif_dir, cif_file)
        io.write(cif_path_out, atoms_snap)
        # Print only first & last snapshot paths for brevity
        if i in (0, len(snapshots)-1):
            print(f"[{base_name} | {T} K]  wrote {cif_file}", flush=True)

    print(f"[DONE] {traj_path}  ({len(snapshots)} frames)", flush=True)

    return dict(
        base_name     = base_name,
        temperature_K = T,
        n_snapshots   = len(snapshots),
        source_path   = cif_path
    )

# ======================================================
# Collect every job (one per temperature)
jobs = []
for root, _, files in os.walk(relaxed_root):
    for fname in files:
        if fname.endswith(".cif") and fname.startswith("cellrelaxed_"):
            cif_path   = os.path.join(root, fname)
            base_name  = os.path.splitext(fname)[0]
            base_meta  = parse_filename_metadata(fname)
            for T in temperatures:
                jobs.append((cif_path, base_name, T, base_meta))

print(f"Total MD jobs to run: {len(jobs)}")

# ------------------------------------------------------
def pool_wrapper(argtuple):
    cif_path, base_name, T, base_meta = argtuple
    result = run_md_task(cif_path, base_name, T)
    result.update(base_meta)
    return result

# ---- safer start-method on GPU / CUDA multiprocessing
try:
    set_start_method("spawn", force=False)
except RuntimeError:
    pass

# You asked for max 2 concurrent temps → processes=2
with Pool(processes=2) as pool:
    results = list(tqdm(pool.imap_unordered(pool_wrapper, jobs),
                        total=len(jobs),
                        desc="MD progress"))

# ---- save metadata ----------------------------------
df = pd.DataFrame(results)
excel_path = os.path.join(md_root, "mdinfo.xlsx")
df.to_excel(excel_path, index=False)

print("\n✅  Multi-temperature MD finished.")
print(f"   Trajectories → {traj_dir}")
print(f"   Snapshots    → {cif_dir}")
print(f"   Metadata     → {excel_path}")
