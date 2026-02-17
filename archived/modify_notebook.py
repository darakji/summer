import json
import os

nb_path = "/home/phanim/harshitrawat/summer/MACE_universal_latent_screening_oods.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

# 1. Add Config to Imports Cell (Index 1)
imports_cell = nb['cells'][1]
config_code = [
    "\n",
    "# Checkpoint Configuration\n",
    "LOAD_FROM_CHECKPOINT = True\n",
    "CHECKPOINT_DIR = \"checkpoints\"\n",
    "os.makedirs(CHECKPOINT_DIR, exist_ok=True)\n",
    "CP_DATA_LOADED = os.path.join(CHECKPOINT_DIR, \"checkpoint_1_data_loaded.pkl\")\n",
    "CP_OOD_RESULTS = os.path.join(CHECKPOINT_DIR, \"checkpoint_2_ood_results.pkl\")\n"
]
imports_cell['source'].extend(config_code)

# 2. Wrap Data Loading Cell (Index 3)
loading_cell = nb['cells'][3]
original_loading_source = loading_cell['source']

new_loading_source = [
    "if LOAD_FROM_CHECKPOINT and os.path.exists(CP_DATA_LOADED):\n",
    "    print(f\"Loading data from checkpoint: {CP_DATA_LOADED}\")\n",
    "    df = pd.read_pickle(CP_DATA_LOADED)\n",
    "    # Re-create X if needed, or save X as well. \n",
    "    # The original code creates X from df['latent']. Let's ensure we do that.\n",
    "    X = np.stack(df['latent'].values)\n",
    "    print(f\"Loaded {len(df)} structures.\")\n",
    "    print(f\"Feature Matrix Shape: {X.shape}\")\n",
    "else:\n"
]

# Indent original code
indented_loading = ["    " + line for line in original_loading_source]
new_loading_source.extend(indented_loading)

# Add saving logic
new_loading_source.extend([
    "\n",
    "    # Save Checkpoint\n",
    "    print(f\"Saving data to checkpoint: {CP_DATA_LOADED}\")\n",
    "    df.to_pickle(CP_DATA_LOADED)\n"
])

loading_cell['source'] = new_loading_source

# 3. Wrap OOD Detection Cell (Index 7)
ood_cell = nb['cells'][7]
original_ood_source = ood_cell['source']

new_ood_source = [
    "if LOAD_FROM_CHECKPOINT and os.path.exists(CP_OOD_RESULTS):\n",
    "    print(f\"Loading OOD results from checkpoint: {CP_OOD_RESULTS}\")\n",
    "    df = pd.read_pickle(CP_OOD_RESULTS)\n",
    "else:\n"
]

# Indent original code
indented_ood = ["    " + line for line in original_ood_source]
new_ood_source.extend(indented_ood)

# Add saving logic
new_ood_source.extend([
    "\n",
    "    # Save Checkpoint\n",
    "    print(f\"Saving OOD results to checkpoint: {CP_OOD_RESULTS}\")\n",
    "    df.to_pickle(CP_OOD_RESULTS)\n"
])

ood_cell['source'] = new_ood_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
