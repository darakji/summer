import os
import pandas as pd
from pymatgen.core.structure import Structure

# Define oxidation state map
ox_states = {"Li": 1, "La": 3, "Zr": 4, "O": -2}

input_dir = "summer_llzo_cifs_prepared"
results = []

for fname in os.listdir(input_dir):
    if fname.endswith(".cif"):
        try:
            struct = Structure.from_file(os.path.join(input_dir, fname))
            
            # Assign oxidation states manually
            try:
                charges = [ox_states[str(site.specie)] for site in struct.sites]
                total_charge = sum(charges)
                dipole_z = sum(q * site.coords[2] for q, site in zip(charges, struct.sites))
                dipole_z = round(dipole_z, 3)
            except Exception as e:
                total_charge = "Failed"
                dipole_z = "Failed"

            results.append((fname, total_charge, dipole_z))
        except Exception as e:
            results.append((fname, "Error", str(e)))

df = pd.DataFrame(results, columns=["File", "Oxidation Sum", "Estimated Dipole Z (eÅ)"])
df.to_excel("llzo_dipole_charge_diagnostics.xlsx", index=False)
print(df)