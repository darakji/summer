
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import seaborn as sns

# Config
DATA_DIR = 'Thesis_Results'
OUTPUT_FILE = os.path.join(DATA_DIR, 'Thesis_Latent_Space_Visualization.png')

def load_pickle(name):
    path = os.path.join(DATA_DIR, name)
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        return pickle.load(f)

def run():
    # 1. Load Data
    # T1_MACE_T1: Training Data (The "Manifold")
    # T3_MACE_T1: OOD Data (The "Outliers")
    df_train = load_pickle('T1_MACE_T1_meta.pkl')
    df_ood = load_pickle('T3_MACE_T1_meta.pkl')
    
    print(f"Train Shape: {df_train.shape}")
    print(f"OOD Shape: {df_ood.shape}")
    
    # 2. Extract Latents
    X_train = np.stack(df_train['latent'].values)
    X_ood = np.stack(df_ood['latent'].values)
    
    # 3. Fit PCA on Training Data Only
    print("Fitting PCA...")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_ood_pca = pca.transform(X_ood)
    
    explained_var = pca.explained_variance_ratio_
    print(f"Explained Variance: {explained_var} (Total: {sum(explained_var):.4f})")
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    OUTPUT_FILE = os.path.join(DATA_DIR, 'Thesis_Latent_Space_Training.png')
    
    # Plot Training Data (Blue)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                c='#005096', alpha=0.3, label='Training Data (T1)', s=15, edgecolors='none')
                
    # Plot OOD Data (Orange) - COMMENTED OUT
    # plt.scatter(X_ood_pca[:, 0], X_ood_pca[:, 1], 
    #             c='#C80000', alpha=0.5, label='OOD Data (T3)', s=15, edgecolors='none')
    
    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontsize=14, fontweight='bold')
    plt.title(f"MACE Latent Space Visualization\n(Learned Manifold)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right', frameon=True, framealpha=0.9)
    
    # Add annotation for total variance
    plt.text(0.05, 0.95, f"Total Variance (2D): {sum(explained_var)*100:.1f}%", 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
             
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved plot to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
