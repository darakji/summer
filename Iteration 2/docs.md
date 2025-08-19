# iteration-2 docs

## Bulk Binary Structure Pull – Materials Project

This iteration focused on pulling **bulk binary compounds** involving the elements **Li, La, Zr, O** directly from the Materials Project database, using the updated `pymatgen` `MPRester.summary_search` interface.

### Query Parameters
- **Binary pairs queried**:
  - Li–O
  - Li–La
  - Li–Zr
  - La–O
  - Zr–O
  - La–Zr
- **Number of elements**: `nelements = 2`
- **Server-side filters**:
  - `is_stable = True`
- **Client-side filters**:
  - Energy above hull ≤ **0.03 eV/atom**
- **Structure type**:
  - Bulk (no supercell, no strain, no perturbation)
  - Saved in CIF format

### Output Summary
| Pair   | Structures Found |
|--------|------------------|
| Li–O   | 3                |
| Li–La  | 0                |
| Li–Zr  | 0                |
| La–O   | 1                |
| Zr–O   | 3                |
| La–Zr  | 0                |
| **Total** | **7**        |

### Output Directory Layout
```

/home/phanim/harshitrawat/summer/binaries\_bulk/
├── index.xlsx                     # Metadata table of all pulled structures
├── pair\_Li\_O/
│   └── cifs/
│       ├── mpid-XXXX\_Li2O\_\_bulk.cif
│       ├── mpid-XXXX\_Li6O\_\_bulk.cif
│       └── ...
├── pair\_La\_O/
│   └── cifs/
│       └── mpid-XXXX\_La2O3\_\_bulk.cif
├── pair\_Zr\_O/
│   └── cifs/
│       ├── mpid-XXXX\_ZrO2\_\_bulk.cif
│       └── mpid-XXXX\_Zr3O\_\_bulk.cif

```

### Notes
- Only **7 total CIFs** were found for the queried constraints.
- No stable structures were returned for Li–La, Li–Zr, or La–Zr within the energy above hull threshold.
- These CIFs serve as **pristine bulk references**; subsequent iterations will add supercells, strain, perturbations, and MD snapshots.
- The `index.xlsx` file includes: `mpid`, `formula`, `chemsys`, stability flag, `energy_above_hull_eV_per_atom`, atom count, and CIF file path.

### Next Steps
- Generate **supercell/strain variants** from these CIFs to expand dataset size.
- Add **elemental references** for Li, La, Zr, O.
- Plan MD runs to increase configuration diversity towards ~3000 total structures.
```
