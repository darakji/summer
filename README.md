# LLZO Slab Preparation Pipeline (CHGNet/MACE Fine-Tuning)

This README documents the current status and progress of LLZO slab preparation for machine learning potential fine-tuning (CHGNet/MACE).

---

## ✅ Project Directory Structure

```
summer/
├── MS_LLZO_surface_data-master/        # Original slab structures from Canepa dataset
├── summer_llzo_cifs/                   # Cleanly renamed CIFs from original .vasp files
├── summer_llzo_cifs_prepared/          # Geometry-prepared CIFs (centered, vacuum added)
├── analyze_dipole_and_charge.py        # Script to compute oxidation sum + rough dipole
├── validate_and_prepare_slabs.py       # Unified slab validator and geometry fixer
├── llzo_dipole_charge_diagnostics.xlsx # Raw oxidation + dipole diagnostics
├── llzo_dipole_charge_tagged.xlsx      # Diagnostics with "Safe"/"Unsafe" labels
├── slab_prep_log.csv                   # Thickness, bond checks, vacuum stats
```

---

## 🔧 Summary of Current Status

### 🔹 Slab Conversion & Cleanup
- All `.vasp` slabs from Canepa repo have been:
  - Renamed cleanly to reflect (hkl), termination, order
  - Converted to `.cif` and stored in `summer_llzo_cifs/`

### 🔹 Validation & Geometry Fixes
- Slabs with:
  - Thickness < 10 Å ✅ Skipped
  - Min bond distance < 1.4 Å ✅ Skipped
- Remaining slabs:
  - ✅ Centered in z-direction
  - ✅ 30 Å vacuum padding added
  - ✅ Stored in `summer_llzo_cifs_prepared/`

### 🔹 Physical Diagnostics
- Oxidation sum calculated using formal charges:
  - ✅ All slabs are charge neutral
- Dipole moments (z) estimated for asymmetry
  - ✅ Tagged as "Safe" or "Unsafe" in `llzo_dipole_charge_tagged.xlsx`

---

## 🔜 Next Steps
- [ ] Exclude Unsafe slabs before CHGNet relaxation
- [ ] Relax slabs using pretrained CHGNet
- [ ] Prepare force/energy labels for MACE

---
