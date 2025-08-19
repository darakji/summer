#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ase import Atoms
from ase.io import read, write
import numpy as np

# -------------------
# 1) Fetch MP Reference Energies
# -------------------
# pip install pymatgen mp-api
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram

API_KEY = "j3J85pX4nLw6asHG9E2lbbCHEKDKgrjc"  # set env var MP_API_KEY instead if you prefer
ELEMENTS = ["Li", "O", "Zr", "La"]

with MPRester(API_KEY) as mpr:
    entries = mpr.get_entries_in_chemsys(ELEMENTS, inc_structure=False)

compat = MaterialsProject2020Compatibility()
proc_entries = compat.process_entries(entries)

pd = PhaseDiagram(proc_entries)

mp_e0s = {el: float(pd.get_reference_energy_per_atom(Composition(el))) for el in ELEMENTS}
print("MP reference energies (eV/atom):", mp_e0s)

# -------------------
# 2) Create isolated-atom frames
# -------------------
frames_iso = []
for el, e0 in mp_e0s.items():
    a = 20.0  # big cubic box to avoid PBC interactions
    atoms = Atoms(symbols=[el], positions=[[0, 0, 0]], cell=[a, a, a], pbc=False)
    atoms.info["REF_energy"] = e0
    atoms.arrays["REF_forces"] = np.zeros((1, 3))
    atoms.info["config_type"] = "IsolatedAtom_MPRef"
    frames_iso.append(atoms)

# -------------------
# 3) Append to existing training file
# -------------------[]
train_file_in = "/home/phanim/harshitrawat/summer/T1_T2_T3_data/d.extxyz"
train_file_out = "/home/phanim/harshitrawat/summer/T1_T2_T3_data/T2_it_2_isolated.extxyz"

frames_train = read(train_file_in, ":")
all_frames = list(frames_train) + frames_iso

write(train_file_out, all_frames, format="extxyz", write_info=True, write_results=True)
print(f"Merged {len(frames_train)} training frames with {len(frames_iso)} isolated-atom frames")
print("Wrote:", train_file_out)
