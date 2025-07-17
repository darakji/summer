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


## 📂 Final Output Characteristics

| Property               | Value                         |
|------------------------|-------------------------------|
| Vacuum (top + bottom)  | 15 Å                           |
| Interface gap (LLZO–Li)| 4 Å                            |
| Thickness mismatch     | ≤ 20% (Phase 1)               |
| Atom count             | 800–1200                      |
| Termination            | Tasker type I (stoichiometric)|
| Output format          | `.cif`, `.json`               |

---

## 🚧 Phase 2 — CHGNet Relaxation of Strain-Matched LLZO‖Li Interfaces

This phase relaxes the heterostructures generated in **Phase 1** using the CHGNet machine-learned potential.

The input structures from `llzo_li_balanced_sliced/` — which are strain-matched, stacked LLZO‖Li slabs — are passed through a multi-stage atomic relaxation workflow using CHGNet and ASE.

---

### 🧘 Relaxation Protocol

- CHGNet v0.4.0 via the ASE interface
- Two-stage relaxation:
  1. `SciPyFminCG` (atomic positions only)
  2. `FIRE` with `relax_cell=True` (to be enabled in the next step of Phase 2)
- Constraints:
  - **Bottom 15 Å** of LLZO and **top 15 Å** of Li are **frozen**
  - **Middle interfacial region** is fully relaxed

Outputs for each structure:
- `relaxed_<structure>.traj`
- `relaxed_<structure>.cif`

---

### ⚙️ Relaxation Notebook Automation

To scale this process, we implemented an automated notebook generator:

- Script: `automate_relaxed.py`
- Scans all `.cif` files in `llzo_li_balanced_sliced/`
- For each:
  - Creates a folder under `relax_final/<structure_name>/`
  - Copies a base Jupyter notebook template
  - Injects a dynamic code cell: `structure_name = "<structure_name>"`

This ensures reproducibility while enabling batch preparation across dozens of structures.

Each notebook is self-contained and directly executable.

---

> 📌 The `relax_demo/` folder contains the prototype notebook that was used as the base template.
> Future updates to this phase will include **cell relaxation and energy-based convergence diagnostics**.

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
