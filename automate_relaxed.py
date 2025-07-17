import os
import shutil
import nbformat

# Configuration
source_cif_folder = "llzo_li_balanced_sliced"
output_base_folder = "relax_final"
base_notebook = "/home/mehuldarak/summer/relax/LLZO_110_Li_order17_off__Li_100_slab_heavy/LLZO_110_Li_order17_off__Li_100_slab_heavy.ipynb"

# Read the base notebook
with open(base_notebook, "r", encoding="utf-8") as f:
    base_nb = nbformat.read(f, as_version=4)

# Collect all CIF files in the source folder
cif_files = [f for f in os.listdir(source_cif_folder) if f.endswith(".cif")]

for cif_file in cif_files:
    structure_name = os.path.splitext(cif_file)[0]

    # Create subfolder in relax/
    out_dir = os.path.join(output_base_folder, structure_name)
    os.makedirs(out_dir, exist_ok=True)

    # Define output notebook path
    out_notebook_path = os.path.join(out_dir, f"{structure_name}.ipynb")

    # Make a deep copy of the base notebook
    new_nb = nbformat.from_dict(base_nb)

    # Insert a new first cell defining `structure_name`
    first_cell = nbformat.v4.new_code_cell(source=f'structure_name = "{structure_name}"')
    new_nb.cells.insert(2, first_cell)

    # Save the new notebook
    with open(out_notebook_path, "w", encoding="utf-8") as f_out:
        nbformat.write(new_nb, f_out)

print("✅ Notebooks generated successfully in 'relax/'")
