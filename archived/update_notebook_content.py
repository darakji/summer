
import nbformat as nbf

nb = nbf.v4.new_notebook()

# 1. Title and Goal
text_intro = """# OOD Latent Space Analysis (Ensemble Uncertainty)

This notebook analyzes the MACE latent embeddings to identify Out-of-Distribution (OOD) atomic environments using **Ensemble Uncertainty**.

## Methodology
We use an ensemble of 4 models ($w_1, w_2, w_3, w_4$) for each training set (T1, T2).

### Metrics
1.  **Ensemble Mean ($\\bar{z}$):** The average embedding vector. We use this "denoised" representation for Density Estimation (GMM).
2.  **Ensemble Variance ($\\sigma^2$):** The average squared Euclidean distance from the mean. This is a direct measure of **Epistemic Uncertainty** (Model Disagreement).

$$ \sigma^2_i = \\frac{1}{4} \\sum_{k=1}^4 ||z_{i,k} - \\bar{z}_i||^2 $$

### Hypothesis
*   **In-Distribution (Reference):** Low Variance, High Density.
*   **Out-of-Distribution (Test):** High Variance (Disagreement) OR Low Density.
"""

# 2. Imports
code_imports = """
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import iread
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-darkgrid')
"""

# 3. Configuration (16 Files)
code_config = """
BASE_DIR = "/home/phanim/harshitrawat/summer/embeddings_results"

files = {
    # MACE T1 (trained on T1)
    "T1_w1_on_T1": "embeddings_MACE_T1_w1_on_T1.extxyz",
    "T1_w2_on_T1": "embeddings_MACE_T1_w2_on_T1.extxyz",
    "T1_w3_on_T1": "embeddings_MACE_T1_w3_on_T1.extxyz",
    "T1_w4_on_T1": "embeddings_MACE_T1_w4_on_T1.extxyz",
    
    "T1_w1_on_T2": "embeddings_MACE_T1_w1_on_T2.extxyz",
    "T1_w2_on_T2": "embeddings_MACE_T1_w2_on_T2.extxyz",
    "T1_w3_on_T2": "embeddings_MACE_T1_w3_on_T2.extxyz",
    "T1_w4_on_T2": "embeddings_MACE_T1_w4_on_T2.extxyz",

    # MACE T2 (trained on T2)
    "T2_w1_on_T2": "embeddings_MACE_T2_w1_on_T2.extxyz",
    "T2_w2_on_T2": "embeddings_MACE_T2_w2_on_T2.extxyz",
    "T2_w3_on_T2": "embeddings_MACE_T2_w3_on_T2.extxyz",
    "T2_w4_on_T2": "embeddings_MACE_T2_w4_on_T2.extxyz",

    "T2_w1_on_T1": "embeddings_MACE_T2_w1_on_T1.extxyz",
    "T2_w2_on_T1": "embeddings_MACE_T2_w2_on_T1.extxyz",
    "T2_w3_on_T1": "embeddings_MACE_T2_w3_on_T1.extxyz",
    "T2_w4_on_T1": "embeddings_MACE_T2_w4_on_T1.extxyz",
}
"""

# 4. Ensemble Load Function
code_load_ensemble = """
def load_ensemble(filenames, limit=None, sample_rate=1, print_interval=1000):
    paths = [os.path.join(BASE_DIR, f) for f in filenames]
    for p in paths:
        if not os.path.exists(p):
            print(f"Error: File not found: {p}")
            return None, None
            
    print(f"Loading Ensemble: {filenames[0]} (and 3 others)...")
    
    mean_list = []
    var_list = []
    
    # Open 4 generators
    gens = [iread(p, index=":") for p in paths]
    
    # Iterate simultaneously
    try:
        for i, atoms_tuple in enumerate(zip(*gens)):
            if limit and i >= limit:
                break
            
            if i % sample_rate == 0:
                # Extract latents: Shape (4, N_atoms, 128)
                latents = []
                valid = True
                for atoms in atoms_tuple:
                    if "mace_latent" not in atoms.arrays:
                        valid = False
                        break
                    latents.append(atoms.arrays["mace_latent"])
                
                if not valid:
                    continue
                    
                # Stack: (4, N, 128)
                stack = np.stack(latents)
                
                # Compute Mean: (N, 128)
                mean = np.mean(stack, axis=0)
                
                # Compute Variance (Scalar per atom): (N,)
                # Sum of squared Euclidean distance from mean, averaged over ensemble
                diff = stack - mean
                norm_sq = np.sum(diff**2, axis=2) # (4, N)
                variance = np.mean(norm_sq, axis=0) # (N,)
                
                mean_list.append(mean)
                var_list.append(variance)
            
            if i > 0 and i % print_interval == 0:
                print(f"Processed {i} frames...", end='\\r')
                
    except Exception as e:
        print(f"Error reading stream: {e}")
        return None, None
        
    print(f"Done. Loaded {len(mean_list)} chunks.")
    
    if not mean_list:
        return np.array([]), np.array([])
        
    X_mean = np.concatenate(mean_list, axis=0)
    X_var = np.concatenate(var_list, axis=0)
    
    print(f"Total Atoms: {len(X_mean)}")
    return X_mean, X_var
"""

# 5. OOD Analysis Logic
code_ood_logic = """
def fit_gmm(ref_data, n_components=5):
    print("Fitting GMM on Reference Data...")
    pca = PCA(n_components=16)
    scaler = StandardScaler()
    
    ref_scaled = scaler.fit_transform(ref_data)
    ref_pca = pca.fit_transform(ref_scaled)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(ref_pca)
    
    return gmm, pca, scaler

def score_gmm(model_tuple, data):
    gmm, pca, scaler = model_tuple
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    # Negative log likelihood (Higher = OOD)
    return -gmm.score_samples(data_pca)
"""

# 6. Analysis T1 Block
code_run_t1 = """
# ANALYSIS 1: MACE_T1 Ensemble
# Trained on T1. Reference = T1. Test = T2.

SAMPLE_RATE = 10 # Adjust for memory

files_ref = [files[f"T1_w{i}_on_T1"] for i in range(1, 5)]
files_test = [files[f"T1_w{i}_on_T2"] for i in range(1, 5)]

print("--- Loading REFERENCE (T1 on T1) ---")
X_mean_ref, X_var_ref = load_ensemble(files_ref, sample_rate=SAMPLE_RATE)

print("--- Loading TEST (T1 on T2) ---")
X_mean_test, X_var_test = load_ensemble(files_test, sample_rate=SAMPLE_RATE)

if X_mean_ref is not None:
    # 1. Uncertainty (Variance) Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(X_var_ref, fill=True, label="Reference (T1)", clip=(0, None))
    sns.kdeplot(X_var_test, fill=True, label="Test (T2)", clip=(0, None))
    plt.title("Model Uncertainty (Ensemble Variance) - MACE T1")
    plt.xlabel("Variance (Uncertainty)")
    plt.legend()
    plt.show()
    
    # 2. Density (GMM) Plot
    models = fit_gmm(X_mean_ref)
    gmm_scores_ref = score_gmm(models, X_mean_ref)
    gmm_scores_test = score_gmm(models, X_mean_test)
    
    plt.figure(figsize=(10, 5))
    sns.kdeplot(gmm_scores_ref, fill=True, label="Reference (T1)")
    sns.kdeplot(gmm_scores_test, fill=True, label="Test (T2)")
    plt.title("Density OOD Score (GMM NLL) - MACE T1")
    plt.xlabel("Negative Log-Likelihood")
    plt.legend()
    plt.show()
"""

# 7. Analysis T2 Block
code_run_t2 = """
# ANALYSIS 2: MACE_T2 Ensemble
# Trained on T2. Reference = T2. Test = T1.

files_ref_2 = [files[f"T2_w{i}_on_T2"] for i in range(1, 5)]
files_test_2 = [files[f"T2_w{i}_on_T1"] for i in range(1, 5)]

print("--- Loading REFERENCE (T2 on T2) ---")
X_mean_ref_2, X_var_ref_2 = load_ensemble(files_ref_2, sample_rate=SAMPLE_RATE)

print("--- Loading TEST (T2 on T1) ---")
X_mean_test_2, X_var_test_2 = load_ensemble(files_test_2, sample_rate=SAMPLE_RATE)

if X_mean_ref_2 is not None:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(X_var_ref_2, fill=True, label="Reference (T2)", clip=(0, None))
    sns.kdeplot(X_var_test_2, fill=True, label="Test (T1)", clip=(0, None))
    plt.title("Model Uncertainty (Ensemble Variance) - MACE T2")
    plt.legend()
    plt.show()
    
    models_2 = fit_gmm(X_mean_ref_2)
    scores_ref_2 = score_gmm(models_2, X_mean_ref_2)
    scores_test_2 = score_gmm(models_2, X_mean_test_2)
    
    plt.figure(figsize=(10, 5))
    sns.kdeplot(scores_ref_2, fill=True, label="Reference (T2)")
    sns.kdeplot(scores_test_2, fill=True, label="Test (T1)")
    plt.title("Density OOD Score (GMM NLL) - MACE T2")
    plt.legend()
    plt.show()
"""

# 8. Diversity Analysis (Clustering High-Uncertainty Points)
code_clustering = """
def analyze_diversity(mean_embeddings, variances, top_n=1000, n_clusters=5):
    print(f"\\n--- Diversity Analysis (Top {top_n} Uncertain Atoms) ---")
    
    # 1. Select High Variance Atoms
    # Note: This is per-atom. For structures, we'd need structure indices. 
    # Here we analyze which *types* of atomic environments are uncertain.
    indices = np.argsort(variances)[-top_n:]
    X_ood = mean_embeddings[indices]
    
    # 2. Cluster them
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_ood)
    
    # 3. Visualize with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_ood)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f"Clustering of Top {top_n} Uncertain Environments")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    
    print("Cluster Counts:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} atoms")

from sklearn.cluster import KMeans

# Run on T2 Test Data
if 'X_mean_test' in locals() and X_mean_test is not None:
    analyze_diversity(X_mean_test, X_var_test, top_n=2000, n_clusters=5)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_code_cell(code_load_ensemble),
    nbf.v4.new_code_cell(code_ood_logic),
    nbf.v4.new_code_cell(code_run_t1),
    nbf.v4.new_code_cell(code_run_t2),
    nbf.v4.new_code_cell(code_clustering)
]

with open('/home/phanim/harshitrawat/summer/OOD_latent.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook recreated with Ensemble Logic + Diversity Clustering.")

