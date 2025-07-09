# LLZO–Li Interface Dataset Generation

This repository contains a complete workflow for constructing high-quality interface slabs between **LLZO (Li₇La₃Zr₂O₁₂)** and **Li metal**. These are intended for use in **machine-learned interatomic potentials (MLIPs)** and **DFT-based interface studies**.

---

## Folder Overview

```bash
summer/
├── letitbeforsometime/             # Temporarily parked code/data (ignore for now)
├── li_slabs_fixed_heavy/           # Finalized Li slabs (cleaned, stoichiometric, no vacuum)
├── llzo_li_balanced_sliced/        # Final LLZO‖Li interface structures (.cif)
├── llzo_li_slabs/                  # Relaxed LLZO surface slabs used for interfacing
├── MS_LLZO_surface_data-master/    # Raw LLZO surface data generation repo (external)
├── summer_llzo_cifs/               # Raw/unprocessed LLZO structures from MP or earlier steps
├── generate_li_slabs.py            # Script to generate and preprocess Li slabs
├── li_llzo_stack.py                # Main script to construct stacked LLZO‖Li interfaces
├── rough.py                        # Utility testing and scratch code
└── README.md                       # ← This documentation
```

---

## Final Interface Structures: `llzo_li_balanced_sliced/`

- All structures are stored in `.cif` format.
- Naming convention:  
  ```
  LLZO_{facet_or_id}__Li_{facet}.cif
  ```

**Key properties:**
- Fully matched XY dimensions
- LLZO and Li block thickness difference ≤ **20%**
- Li placed with **4 Å interfacial gap**
- **Tasker Type I surfaces** (dipole-free, stoichiometric)
- 15 Å vacuum above and below
- Atom counts: **800–1200 per structure**

---

## Workflow Summary

### 1. `generate_li_slabs.py`

- Creates Li surface slabs along low-index facets
- Optional heavy-atom fixes and vacuum removal
- Final slabs stored in `li_slabs_fixed_heavy/`

### 2. `li_llzo_stack.py` (Main Interface Builder)

For each LLZO–Li slab pair:

- Li is **tiled in X and Y** to match LLZO lateral dimensions
- For a range of Z-repeats \(k = 1 \dots 8\):
  - Li block is stacked
  - Top part is **sliced** at LLZO thickness to ensure close match
- Final selection is based on:
  - Thickness mismatch \( < 20\% \)
  - Atom count between **800 and 1200**
  - Score function balancing both:

    \[
    \text{score} = \frac{|\Delta t|}{t_{\text{LLZO}}} + \left| \frac{N_{\text{Li}} - 1000}{1000} \right|
    \]

- LLZO is shifted to 15 Å (Z), Li stacked above with 4 Å gap
- Final cell height = top of Li + 15 Å vacuum

---

## Design Considerations

- Interfaces are physically realistic and balanced
- All slabs are neutral and dipole-free
- Designed for CHGNet, MACE, DFT-FE, and MD workflows
- Atom count optimized for parallel computation

---

## Utility Scripts

| Script               | Purpose                                      |
|----------------------|----------------------------------------------|
| `generate_li_slabs.py` | Generate Li facets from bulk from MP's cif |
| `li_llzo_stack.py`     | Main stacking & slicing logic |
| `rough.py`             | Prototyping and experimental code |

---

## Folder: `letitbeforsometime/`

This contains temporary or archived scripts/data not currently active.  
→ **Parked for future consideration**, safe to ignore in current workflow.

---

## Requirements

- `ase`
- `pymatgen`
- `numpy`

Add optional: `spglib`, `ovito` for visualization/export.

---

## Applications

- Formation energy of interfaces
- Interface relaxation studies
- Electrochemical stability
- Lithium penetration modeling
- MLIP fine-tuning for heterostructures

---

## Contributions & Help

Feel free to fork, contribute or raise issues.
