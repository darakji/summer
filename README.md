# LLZO–Li Interface Dataset Generation

This repository provides a rigorous and modular workflow for constructing realistic interface slabs between **LLZO (Li₇La₃Zr₂O₁₂)** and **Li metal**, designed for use in:

- Machine-learned interatomic potentials (MLIPs)
- DFT-based relaxation and interfacial energy studies
- Molecular dynamics simulations

---

## 📁 Folder Overview

```bash
summer/
├── letitbeforsometime/             # Parked/archived logic (not currently in use)
├── li_slabs_fixed_heavy/           # Final Li slabs (vacuum-free, stoichiometric)
├── llzo_li_balanced_sliced/        # Phase 1 strain-matched interface structures
├── llzo_li_slabs/                  # Relaxed LLZO surface slabs (multiple facets)
├── summer_llzo_cifs/               # Raw LLZO input structures from Canepa et al. 2018 paper
├── generate_li_slabs.py            # Builds Li surface slabs from bulk
├── li_llzo_stack.py                # Phase 1 stacking + slicing script
├── relax_demo       # Phase 2 Relaxing with CHGNet
├── rough.py                        # Utility scratchpad for prototyping
└── README.md                       # ← This documentation
```

---

## 🧩 Phase 1 — Strain-Matched LLZO‖Li Interfaces

This phase constructs LLZO‖Li stacks using **XY tiling and Z-slicing**, optionally straining Li to match LLZO dimensions.

### Steps:
1. **Slab Preparation**:
   - Input LLZO and Li slabs are vacuum-stripped and stoichiometric.
   - All slabs are terminated as **Tasker type I** (dipole-free, charge neutral).

2. **XY Tiling**:
   - Li slab is **tiled in X and Y** to cover LLZO in-plane dimensions.
   - Li is **stretched isotropically** in-plane to match LLZO cell:

     $
     \mathbf{a}_{\text{Li}}' = \frac{L_{\text{LLZO}}^x}{L_{\text{Li}}^x} \cdot \mathbf{a}_{\text{Li}}, \quad
     \mathbf{b}_{\text{Li}}' = \frac{L_{\text{LLZO}}^y}{L_{\text{Li}}^y} \cdot \mathbf{b}_{\text{Li}}
     $

3. **Z-stacking and Slicing**:
   - For each Z-repeat factor $k \in [1, 8]$, Li block is sliced at height t_{\text{LLZO}}.
   - Candidate slab is accepted if:
     - Atom count $\in [800, 1200]$
     - Relative thickness mismatch $\delta_t < 20\%$:

       $
       \delta_t = \left|\frac{t_{\text{Li}} - t_{\text{LLZO}}}{t_{\text{LLZO}}}\right|
       $

4. **Scoring (if no exact match)**:

   $
   \text{score} = \delta_t + \left| \frac{N_{\text{Li}} - 1000}{1000} \right|
   $

5. **Stacking & Output**:
   - LLZO is shifted to start at 15 Å (bottom vacuum).
   - Li block stacked above LLZO + 4 Å interface gap.
   - Top vacuum: 15 Å → total Z height:
   
     $
     L_z = 15 + t_{\text{LLZO}} + 4 + t_{\text{Li}} + 15
     $

- 📁 All resulting `.cif` structures are saved in `llzo_li_balanced_sliced/`.

---

## 📂 Final Output Characteristics

| Property               | Value                         |
|------------------------|-------------------------------|
| Vacuum (top + bottom)  | 15 Å                           |
| Interface gap (LLZO–Li)| 4 Å                            |
| Thickness mismatch     | ≤ 20% (Phase 1)               |
| Lattice mismatch       | ≤ 5% (Phase 2)                |
| Atom count             | 800–1200                      |
| Termination            | Tasker type I (stoichiometric)|
| Output format          | `.cif`, `.json`               |

---

## 🚧 Phase 2 — Relaxing Structures with CHGNet
# 📂 relax_demo

This folder contains the full CHGNet-based relaxation workflow and results for a **single LLZO‖Li interface** structure.

---

## 📘 LLZO‖Li Interface Relaxation Notebook

Each notebook in this series handles **only one structure**.  
This one corresponds to:

- **Structure**: `LLZO_001_Zr_code93_sto__Li_110_slab_heavy`
- **Initial lattice height**: 74.46 Å
- **Number of atoms**: 900

---

### 1. Purpose
- Relax the LLZO‖Li heterostructure using CHGNet
- Perform multi-stage optimization (CG → FIRE)
- Freeze bulk-like regions (15 Å at both ends)
- Relax the lattice vectors to relieve interfacial strain

---

### 2. Method
- **CHGNet v0.4.0** with ASE interface
- Stage 1: `SciPyFminCG` (no cell relaxation) → fₘₐₓ ≈ 0.15 eV/Å
- Stage 2: `FIRE` with `relax_cell=True` → fₘₐₓ < 0.05 eV/Å

---

### 3. Constraints
- **LLZO base**: bottom 15 Å frozen
- **Li top**: top 15 Å frozen
- **Interfacial region** relaxed in all directions

---

### 4. Outputs
- `relaxed_LLZO_Li_interface_15A_frozen.cif`
- `relaxed_LLZO_Li_interface_15A_frozen.traj`
- `relaxation.traj` (intermediate trajectory)
- `relaxation_log.pkl` (optional log with energy, force data)

---

### 5. Visual + Structural Checks
- Pre/post relaxation visualizations (To be updated soon)
- Z-analysis to confirm no Li diffusion into LLZO
- Final force convergence: **fₘₐₓ ≈ 0.043 eV/Å**

---

### 🧭 What's Next?

We will be **extending this workflow to other LLZO terminations and Li slab orientations**, following the same methodology and documentation format — one structure per notebook.

---

## 📚 References

1. Kostiantyn V. Kravchyk, Huanyu Zhang & Maksym V. Kovalenko, On the interfacial phenomena at the Li₇La₃Zr₂O₁₂ (LLZO)/Li interface, Communications Chemistry 7, 257 (2024)
➡️ https://doi.org/10.1038/s42004-024-01350-9 
American Chemical Society Publications Nature Research Collection

2. Li–Garnet Solid‑State Batteries with LLZO Scaffolds, Accounts of Materials Research 3, 1–12 (2022)—discusses coherent lattice interfacial matches in LLZO systems
➡️ https://doi.org/10.1021/acsenergylett.2c00004 
Taylor & Francis Online. American Chemical Society Publications

3. Controlling the lithium proton exchange of LLZO…, J. Mater. Chem. A 9, 4831–4840 (2021) — covers LLZO surface preparation and key pre-treatment methods
➡️ https://doi.org/10.1039/D0TA11096E 
PMC RSC Publishing SciSpace

4. **ASE Build Docs**, *Building Slabs with Miller Indices*  
   🔗 [https://wiki.fysik.dtu.dk/ase/ase/build/build.html](https://wiki.fysik.dtu.dk/ase/ase/build/build.html)

5. **MACE Docs**, *Fine-Tuning Foundation Models*  
   🔗 [https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)

---

## 🔧 Dependencies

- Python 3.9+
- `ase`
- `pymatgen`
- `numpy`
- (Optional) `spglib`, `ovito`, `chgnet`, `mace`, `avogadro`

---

## ✨ Applications

- CHGNet / MACE pre-relaxation
- Formation energy benchmarking
- Li interface diffusion modeling
- Dataset generation for MLIP fine-tuning
- Strain vs registry studies

---

## 🙌 Contributions & Contact

If you'd like to contribute to coherent matching, automation, or CHGNet fine-tuning — feel free to fork, open an issue, or contact the maintainer.
