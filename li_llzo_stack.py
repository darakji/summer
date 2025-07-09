import os, numpy as np
from glob import glob
from ase.io import read, write
from ase.build import make_supercell, sort

# ─── User Settings ───
LLZO_DIR     = "llzo_li_slabs"
LI_DIR       = "li_slabs_fixed_heavy"
OUT_DIR      = "llzo_li_balanced_sliced"

GAP          = 4.0        # Å between LLZO & Li
VAC_TOP      = 15.0       # Å vacuum above Li
ATOM_MIN     = 800
ATOM_MAX     = 1200
PARITY_TOL   = 0.20       # ±20% thickness tolerance
MAX_LAYERS   = 8          # max Li repeats to try
# ──────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

def strip_vacuum(slab):
    zmin, zmax = slab.positions[:,2].min(), slab.positions[:,2].max()
    thick = zmax - zmin
    cell  = slab.get_cell().copy()
    cell[2,2] = thick
    slab.set_cell(cell, scale_atoms=False)
    slab.positions[:,2] -= zmin
    return thick

def tile_xy(li, llzo):
    # true in-plane vectors
    v1, v2 = li.cell[0].copy(), li.cell[1].copy()
    v1[2]=v2[2]=0
    a_li, b_li = np.linalg.norm(v1), np.linalg.norm(v2)
    a_L, b_L   = llzo.cell.lengths()[:2]
    n1, n2     = int(np.ceil(a_L/a_li)), int(np.ceil(b_L/b_li))
    li_big = make_supercell(li, np.diag([n1,n2,1]))
    # orthogonalize to a_L × b_L
    c = li_big.cell[2,2]
    li_big.set_cell([[a_L,0,0],[0,b_L,0],[0,0,c]], scale_atoms=True)
    return li_big

def slice_and_score(li_block, t_LLZO):
    # slice at z=t_LLZO
    mask = li_block.positions[:,2] <= t_LLZO + 1e-6
    slab = li_block[mask]
    # recalc thickness
    zmax = slab.positions[:,2].max()
    natoms = len(slab)
    # compute scores
    dt = abs(zmax - t_LLZO)/t_LLZO
    da = abs(natoms - 1000)/1000
    return slab, zmax, natoms, dt, da

# ── Main ──
for llzo_path in sorted(glob(f"{LLZO_DIR}/*.cif")):
    for li_path in sorted(glob(f"{LI_DIR}/*.cif")):
        name_LL = os.path.splitext(os.path.basename(llzo_path))[0]
        name_LI = os.path.splitext(os.path.basename(li_path))[0]
        try:
            llzo = read(llzo_path); li = read(li_path)
            t_LL = strip_vacuum(llzo)
            strip_vacuum(li)
            li_xy = tile_xy(li, llzo)

            best = None
            for k in range(1, MAX_LAYERS+1):
                block = make_supercell(li_xy, np.diag([1,1,k]))
                slab, t_LI, nat, dt, da = slice_and_score(block, t_LL)
                ok_thick = dt <= PARITY_TOL
                ok_atoms = ATOM_MIN <= nat <= ATOM_MAX
                score = dt + da
                # first acceptable
                if ok_thick and ok_atoms:
                    best = (slab, t_LI, nat)
                    print(f"→ {name_LI}: k={k} PASS (t={t_LI:.2f}, atoms={nat})")
                    break
                # track best if none pass
                if best is None or score < best[3]:
                    best = (slab, t_LI, nat, score, k)
            # if we fell out without perfect pass
            if not (ATOM_MIN <= best[2] <= ATOM_MAX and abs(best[1]-t_LL)/t_LL<=PARITY_TOL):
                slab, t_LI, nat, score, k = best
                print(f"⚠ {name_LI}: best k={k} (t={t_LI:.2f}, atoms={nat}, score={score:.2f})")

            # stack
            llzo.positions[:,2] += VAC_TOP
            slab.positions[:,2] += VAC_TOP + t_LL + GAP

            combo = sort(llzo + slab)
            # final cell height
            ztop = combo.positions[:,2].max()
            c = combo.get_cell().copy(); c[2,2] = ztop + VAC_TOP
            combo.set_cell(c, scale_atoms=False)
            # center XY
            mins = combo.positions[:,:2].min(0); maxs = combo.positions[:,:2].max(0)
            center = 0.5*(mins+maxs)
            combo.positions[:,:2] -= (center - 0.5*combo.get_cell().lengths()[:2])

            out = f"{OUT_DIR}/{name_LL}__{name_LI}.cif"
            write(out, combo)
            print(f"✅ {name_LL}__{name_LI}: atoms={len(combo)}, LLZO_t={t_LL:.2f}, LI_t={best[1]:.2f}")

        except Exception as e:
            print(f"❌ {name_LL}×{name_LI} → {e}")
