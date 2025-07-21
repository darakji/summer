import json
import pandas as pd
import torch
from ase.io import read
from ase import Atoms
from mace.data import AtomicData
from mace.modules import MACE
from mace.calculators import MACECalculator
from torch.utils.data import Dataset, DataLoader
import numpy as np

# === Load CIF ===
cif_path = "/home/phanim/harshitrawat/summer/summer_llzo_cifs/LLZO_010_La_order0_off.cif"
atoms = read(cif_path)

# === Load CHGNet-predicted forces from Excel ===
df = pd.read_excel("/home/phanim/harshitrawat/summer/chgnet_single_test_output.xlsx")
row = df[df["file"] == "LLZO_010_La_order0_off.cif"].iloc[0]

forces = np.array(json.loads(row["forces_per_atom_eV_per_A"]))
energy = float(row["energy_eV"])

# === Inject forces and energy into ASE Atoms object ===
atoms.calc = None  # no external calculator
atoms.info["energy"] = energy
atoms.arrays["forces"] = forces

# === Wrap into MACE dataset ===
class SingleAtomsDataset(Dataset):
    def __init__(self, atoms_obj):
        self.atoms = atoms_obj

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = AtomicData.from_ase(self.atoms)
        data.y = torch.tensor([self.atoms.info["energy"]], dtype=torch.float32)
        data.forces = torch.tensor(self.atoms.arrays["forces"], dtype=torch.float32)
        return data

dataset = SingleAtomsDataset(atoms)
dataloader = DataLoader(dataset, batch_size=1)

# === Define a small MACE model (CPU-friendly) ===
model = MACE(
    r_max=5.0,
    num_bessel=6,
    num_polynomial_cutoff=5,
    hidden_irreps="128x0e + 128x1o",
    num_interactions=2,
    max_ell=2,
    correlation=2,
    atomic_energies={"Li": 0.0, "La": 0.0, "Zr": 0.0, "O": 0.0},  # dummy values
    avg_num_neighbors=10.0,
    scale_file=None,
)

model.to("cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Minimal Training Loop ===
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(batch)
        loss_e = torch.nn.functional.mse_loss(pred["energy"], batch.y)
        loss_f = torch.nn.functional.mse_loss(pred["forces"], batch.forces)
        loss = loss_e + loss_f
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.6f} (E={loss_e.item():.4f}, F={loss_f.item():.4f})")

# === Inference on same structure ===
model.eval()
with torch.no_grad():
    pred = model(AtomicData.from_ase(atoms))
    print("\n🔍 Predicted Energy:", pred["energy"].item())
    print("🔍 Predicted Forces (first 3 atoms):\n", pred["forces"][:3].numpy())

# === Save MACE predictions to Excel ===
output_excel = "mace_predictions_test.xlsx"
mace_energy = pred["energy"].item()
mace_forces = pred["forces"].cpu().numpy().tolist()  # shape (N_atoms, 3)

# Prepare DataFrame
df_out = pd.DataFrame({
    "atom_index": list(range(len(mace_forces))),
    "fx": [f[0] for f in mace_forces],
    "fy": [f[1] for f in mace_forces],
    "fz": [f[2] for f in mace_forces],
})
df_out["structure"] = "LLZO_010_La_order0_off.cif"
df_out["mace_energy_eV"] = mace_energy

# Save
df_out.to_excel(output_excel, index=False)
print(f"\n📝 MACE predictions saved to: {output_excel}")
