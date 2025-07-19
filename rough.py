from ase.io import Trajectory
from chgnet.model.dynamics import CHGNetCalculator

# Load one of your traj files
traj = Trajectory("md/mdtraj/cellrelaxed_LLZO_010_Li_order0_off__Li_111_slab_heavy_T300.traj")
atoms = traj[0]

# Attach CHGNet calculator
atoms.calc = CHGNetCalculator()

# Force CHGNet to evaluate everything
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"Energy: {energy:.4f} eV")
print(f"fmax: {max((forces**2).sum(axis=1)**0.5):.4f} eV/Å")
print(f"Stress: {stress}")
