import json
import os
import re

nb_path = "/home/phanim/harshitrawat/summer/MACE_universal_latent_screening_oods.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

# --- 1. Clean Imports Cell (Index 1) ---
# The imports cell (index 1) has duplicate config.
imports_cell = nb['cells'][1]
source = imports_cell['source']
# We want to keep the imports and ONE config block.
# The config block starts with "# Checkpoint Configuration"
# Let's find all occurrences.
indices = [i for i, line in enumerate(source) if "# Checkpoint Configuration" in line]
if len(indices) > 0:
    # Keep everything up to the first config block + its content (approx 7 lines)
    # Config block:
    # # Checkpoint Configuration
    # LOAD_FROM_CHECKPOINT = True
    # CHECKPOINT_DIR = "checkpoints"
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # CP_DATA_LOADED = ...
    # CP_OOD_RESULTS = ...
    # (newline)
    
    first_idx = indices[0]
    # We'll keep 7 lines for the config.
    # But we should check if there are other things after the duplicates? 
    # Usually imports are at the top.
    # Let's just truncate after the first config block.
    imports_cell['source'] = source[:first_idx+7]

# --- 2. Clean Data Loading Cell (Index 3) ---
loading_cell = nb['cells'][3]
original_source = loading_cell['source']

# We need to extract the core logic (parsing filenames, looping files) which is currently deeply nested.
# Pattern to find the core logic: look for "def parse_filename"
core_start = -1
for i, line in enumerate(original_source):
    if "def parse_filename(filename):" in line:
        core_start = i
        break

if core_start != -1:
    # Determine indentation of the core logic
    first_line = original_source[core_start]
    indentation = len(first_line) - len(first_line.lstrip())
    indent_str = first_line[:indentation]
    
    # Extract and dedent
    cleaned_logic = []
    for line in original_source[core_start:]:
        # Stop if we hit the "Save Checkpoint" lines which might be at different indentation or just at the end
        if "# Save Checkpoint" in line:
            break
        
        if line.startswith(indent_str):
            cleaned_logic.append(line[indentation:])
        elif line.strip() == "":
            cleaned_logic.append(line)
        else:
            # If line is less indented, it might be outside the block we want, but inside the cell.
            # In the nested structure, the core logic is the most indented.
            # So anything less indented is likely the `else:` or `if` wrappers we want to discard.
            pass

    # Construct new cell source
    new_source = [
        "if LOAD_FROM_CHECKPOINT and os.path.exists(CP_DATA_LOADED):\n",
        "    print(f\"Loading data from checkpoint: {CP_DATA_LOADED}\")\n",
        "    df = pd.read_pickle(CP_DATA_LOADED)\n",
        "    # Re-create X if needed\n",
        "    X = np.stack(df['latent'].values)\n",
        "    print(f\"Loaded {len(df)} structures.\")\n",
        "    print(f\"Feature Matrix Shape: {X.shape}\")\n",
        "else:\n"
    ]
    
    for line in cleaned_logic:
        new_source.append("    " + line)
        
    new_source.extend([
        "\n",
        "    # Save Checkpoint\n",
        "    print(f\"Saving data to checkpoint: {CP_DATA_LOADED}\")\n",
        "    df.to_pickle(CP_DATA_LOADED)\n"
    ])
    
    loading_cell['source'] = new_source

# --- 3. Clean OOD Cell (Index 7) ---
ood_cell = nb['cells'][7]
original_ood = ood_cell['source']

# Core logic starts with defining X_train or the comment before it
core_ood_start = -1
for i, line in enumerate(original_ood):
    if "# We define OOD based on the Training Set" in line:
        core_ood_start = i
        break

if core_ood_start != -1:
    first_line = original_ood[core_ood_start]
    indentation = len(first_line) - len(first_line.lstrip())
    indent_str = first_line[:indentation]
    
    cleaned_ood = []
    for line in original_ood[core_ood_start:]:
        if "# Save Checkpoint" in line:
            break
        if line.startswith(indent_str):
            cleaned_ood.append(line[indentation:])
        elif line.strip() == "":
            cleaned_ood.append(line)
            
    new_ood_source = [
        "if LOAD_FROM_CHECKPOINT and os.path.exists(CP_OOD_RESULTS):\n",
        "    print(f\"Loading OOD results from checkpoint: {CP_OOD_RESULTS}\")\n",
        "    df = pd.read_pickle(CP_OOD_RESULTS)\n",
        "else:\n"
    ]
    
    for line in cleaned_ood:
        new_ood_source.append("    " + line)
        
    new_ood_source.extend([
        "\n",
        "    # Save Checkpoint\n",
        "    print(f\"Saving OOD results to checkpoint: {CP_OOD_RESULTS}\")\n",
        "    df.to_pickle(CP_OOD_RESULTS)\n"
    ])
    
    ood_cell['source'] = new_ood_source

# --- 4. Update PCA Cell (Index 5) ---
pca_cell = nb['cells'][5]
pca_code = [
    "\n",
    "# Standardize\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# PCA\n",
    "pca = PCA(n_components=50) # Check top 50\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "\n",
    "# Cumulative Variance\n",
    "cum_var = np.cumsum(pca.explained_variance_ratio_)\n",
    "\n",
    "# Plot Cumulative Variance\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(range(1, 51), cum_var, marker='o', linestyle='--')\n",
    "plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance')\n",
    "plt.axhline(y=0.95, color='g', linestyle=':', label='95% Variance')\n",
    "plt.axhline(y=0.99, color='k', linestyle=':', label='99% Variance')\n",
    "plt.title('Cumulative Variance vs Dimensions')\n",
    "plt.xlabel('Number of Components')\n",
    "plt.ylabel('Cumulative Explained Variance')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Print Thresholds\n",
    "n_90 = np.argmax(cum_var >= 0.90) + 1\n",
    "n_95 = np.argmax(cum_var >= 0.95) + 1\n",
    "n_99 = np.argmax(cum_var >= 0.99) + 1\n",
    "\n",
    "print(f\"Dimensions needed for 90% variance: {n_90}\")\n",
    "print(f\"Dimensions needed for 95% variance: {n_95}\")\n",
    "print(f\"Dimensions needed for 99% variance: {n_99}\")\n",
    "\n",
    "# Table for first 10\n",
    "print(\"\\nDimensions vs Cumulative Variance (First 10):\")\n",
    "for i in range(10):\n",
    "    print(f\"Dim {i+1}: {cum_var[i]:.4f}\")\n",
    "\n",
    "# Add PC1/PC2 to DataFrame\n",
    "df['PC1'] = X_pca[:, 0]\n",
    "df['PC2'] = X_pca[:, 1]\n",
    "\n",
    "# Plot Latent Space\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.scatterplot(data=df, x='PC1', y='PC2', hue='dataset', alpha=0.6, palette='viridis')\n",
    "plt.title('Universal Latent Space (256D -> 2D)')\n",
    "plt.show()\n"
]
pca_cell['source'] = pca_code

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
