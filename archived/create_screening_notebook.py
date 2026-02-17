
import nbformat as nbf

nb = nbf.v4.new_notebook()

# ------------------------------------------------------------------------------
# 1. Imports & Setup
# ------------------------------------------------------------------------------
cell_imports = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ase.io
import re
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import hdbscan

# Plotting Style
sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
DATA_DIR = "universal_embeddings_results"
FILES = {
    "T1": os.path.join(DATA_DIR, "Universal_on_T1.xyz"),
    "T2": os.path.join(DATA_DIR, "Universal_on_T2.xyz"),
    "T3": os.path.join(DATA_DIR, "Universal_on_T3.xyz")
}
"""

# ------------------------------------------------------------------------------
# 2. Data Loading & Parsing
# ------------------------------------------------------------------------------
cell_load = """
def parse_filename(filename):
    # Regex based on user convention:
    # cellrelaxed_LLZO_{cleavingdir}_{termination}_{order}_{sto|offsto}__Li_{facet}_slab_heavy_T{Temp}_{Index}.cif
    # Also handling strain: ..._strain{+/-}{val}_perturbed.cif
    
    meta = {}
    meta['filename'] = filename
    
    # Strain
    strain_match = re.search(r"strain([+-]?[\d\.]+)_perturbed", filename)
    if strain_match:
        meta['strain'] = float(strain_match.group(1))
        meta['is_perturbed'] = True
    else:
        meta['strain'] = 0.0
        meta['is_perturbed'] = False

    # Temperature
    temp_match = re.search(r"_T(\d+)_", filename)
    if temp_match:
        meta['temp'] = int(temp_match.group(1))
    else:
        meta['temp'] = None

    # Facet (e.g., Li_100_slab)
    facet_match = re.search(r"Li_(\d+)_slab", filename)
    if facet_match:
        meta['facet'] = facet_match.group(1)
    else:
        meta['facet'] = "Unknown"

    # Termination (e.g., LLZO_010_La_order0)
    # This is tricky, let's try to capture the block between LLZO_ and __Li
    term_match = re.search(r"LLZO_(.*?)__Li", filename)
    if term_match:
        parts = term_match.group(1).split('_')
        # Heuristic: usually {cleaving}_{termination}_{order}_{sto}
        if len(parts) >= 2:
            meta['termination'] = parts[1] # e.g. La
        else:
            meta['termination'] = "Unknown"
    else:
        meta['termination'] = "Unknown"
        
    return meta

data_list = []

for dataset_name, filepath in FILES.items():
    print(f"Loading {dataset_name} from {filepath}...")
    atoms_list = ase.io.read(filepath, index=":")
    
    for atoms in atoms_list:
        info = atoms.info
        arrays = atoms.arrays
        
        # Extract Latent (256D)
        # Note: 'mace_latent' is per-atom. We need a global descriptor.
        # Strategy: MEAN of atomic latents.
        if 'mace_latent' in arrays:
            latent = np.mean(arrays['mace_latent'], axis=0) # Shape (256,)
        else:
            continue
            
        # Extract Energy
        energy = info.get('mace_energy', np.nan)
        
        # Parse Filename (stored in info or we assume order?)
        # MACE usually preserves info. Let's assume 'filename' or 'comment' holds it.
        # If not, we might need to rely on index if filenames weren't saved.
        # CHECK: The user's extraction script likely saved filenames in info if they were in the input.
        # If input was .extxyz, it might have 'config_type' or similar.
        # Let's assume there is a way to identify. For now, we use a placeholder if missing.
        fname = info.get('filename', info.get('comment', f"unknown_{dataset_name}"))
        
        entry = parse_filename(fname)
        entry['dataset'] = dataset_name
        entry['energy'] = energy
        entry['latent'] = latent
        
        data_list.append(entry)

df = pd.DataFrame(data_list)
print(f"Loaded {len(df)} structures.")

# Create Feature Matrix X
X = np.stack(df['latent'].values)
print(f"Feature Matrix Shape: {X.shape}")
"""

# ------------------------------------------------------------------------------
# 3. PCA & Scree Plot
# ------------------------------------------------------------------------------
cell_pca = """
# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=50) # Check top 50
X_pca = pca.fit_transform(X_scaled)

# Scree Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Scree Plot: Variance vs Dimensions')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

cum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"Cumulative Variance (PC1+PC2): {cum_var[1]:.4f}")

# Add PC1/PC2 to DataFrame
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Plot Latent Space
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='dataset', alpha=0.6, palette='viridis')
plt.title('Universal Latent Space (256D -> 2D)')
plt.show()
"""

# ------------------------------------------------------------------------------
# 4. OOD Detection (LOF, KNN, HDBSCAN)
# ------------------------------------------------------------------------------
cell_ood = """
# We define OOD based on the Training Set (T1) distribution.
X_train = X_scaled[df['dataset'] == 'T1']

# 1. KNN Distance
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_train)
distances, _ = knn.kneighbors(X_scaled)
df['knn_dist'] = np.mean(distances, axis=1)

# 2. LOF (Local Outlier Factor)
# Note: LOF is transductive (fit_predict), but we can use decision_function on new data if novelty=True
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train)
df['lof_score'] = -lof.decision_function(X_scaled) # Negate so higher is more outlier

# 3. HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
labels = clusterer.fit_predict(X_scaled)
df['hdbscan_cluster'] = labels
df['hdbscan_outlier_score'] = clusterer.outlier_scores_

# Thresholds (99th percentile of T1)
knn_thresh = np.percentile(df[df['dataset']=='T1']['knn_dist'], 99)
lof_thresh = np.percentile(df[df['dataset']=='T1']['lof_score'], 99)

df['is_ood_knn'] = df['knn_dist'] > knn_thresh
df['is_ood_lof'] = df['lof_score'] > lof_thresh
df['is_ood_hdbscan'] = df['hdbscan_cluster'] == -1

print(f"KNN Threshold: {knn_thresh:.4f}")
print(f"LOF Threshold: {lof_thresh:.4f}")
"""

# ------------------------------------------------------------------------------
# 5. Bucketing & Analysis
# ------------------------------------------------------------------------------
cell_bucket = """
# Define Buckets
df['bucket_strain'] = df['strain'].apply(lambda x: f"Strain {x}")
df['bucket_term'] = df['termination']
df['bucket_facet'] = df['facet']

# Analysis Function
def analyze_ood_rates(group_col, method_col):
    stats = df.groupby(group_col)[method_col].agg(['count', 'sum', 'mean']).reset_index()
    stats.columns = [group_col, 'Total', 'OOD_Count', 'OOD_Rate']
    return stats.sort_values('OOD_Rate', ascending=False)

print("--- OOD Analysis by Strain (KNN) ---")
print(analyze_ood_rates('bucket_strain', 'is_ood_knn'))

print("\\n--- OOD Analysis by Termination (KNN) ---")
print(analyze_ood_rates('bucket_term', 'is_ood_knn'))

# Venn Diagram / Overlap
ood_knn = set(df[df['is_ood_knn']].index)
ood_lof = set(df[df['is_ood_lof']].index)
ood_hdb = set(df[df['is_ood_hdbscan']].index)

overlap = ood_knn.intersection(ood_lof).intersection(ood_hdb)
print(f"\\nStructures flagged by ALL three methods: {len(overlap)}")
"""

# Add cells
nb.cells.append(nbf.v4.new_markdown_cell("# Universal Latent Space Screening"))
nb.cells.append(nbf.v4.new_code_cell(cell_imports))
nb.cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading & Regex Parsing"))
nb.cells.append(nbf.v4.new_code_cell(cell_load))
nb.cells.append(nbf.v4.new_markdown_cell("## 2. PCA & Scree Plot"))
nb.cells.append(nbf.v4.new_code_cell(cell_pca))
nb.cells.append(nbf.v4.new_markdown_cell("## 3. OOD Detection (KNN, LOF, HDBSCAN)"))
nb.cells.append(nbf.v4.new_code_cell(cell_ood))
nb.cells.append(nbf.v4.new_markdown_cell("## 4. Metadata Bucketing Analysis"))
nb.cells.append(nbf.v4.new_code_cell(cell_bucket))

# Write to file
with open('summer/MACE_universal_latent_screening_oods.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated: summer/MACE_universal_latent_screening_oods.ipynb")
