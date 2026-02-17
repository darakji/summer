#!/usr/bin/env python3
"""Latent space screening pipeline converted from notebook.

Features:
- Load latent vectors from .extxyz / ASE-readable files (expects per-atom 'mace_latent')
- Build metadata, checkpointing
- PCA: find n_components preserving >=99% variance, visualize
- LOF and KNN anomaly scores (parallelized)
- Metadata analysis and outputs
"""
import os
import re
import argparse
import logging
from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd
from ase import io
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


def parse_filename(filename):
    meta = {}
    meta['filename'] = filename
    strain_match = re.search(r"strain([+-]?[\d\.]+)_perturbed", filename)
    if strain_match:
        meta['strain'] = float(strain_match.group(1))
        meta['is_perturbed'] = True
    else:
        meta['strain'] = 0.0
        meta['is_perturbed'] = False
    temp_match = re.search(r"_T(\d+)_", filename)
    meta['temp'] = int(temp_match.group(1)) if temp_match else None
    facet_match = re.search(r"Li_(\d+)_slab", filename)
    meta['facet'] = facet_match.group(1) if facet_match else "Unknown"
    term_match = re.search(r"LLZO_(.*?)__Li", filename)
    if term_match:
        parts = term_match.group(1).split('_')
        meta['termination'] = parts[1] if len(parts) >= 2 else "Unknown"
    else:
        meta['termination'] = "Unknown"
    return meta


def load_files(label_path_pairs, checkpoint=None, load_from_checkpoint=False):
    if load_from_checkpoint and checkpoint and os.path.exists(checkpoint):
        logging.info(f"Loading checkpoint {checkpoint}")
        df = pd.read_pickle(checkpoint)
        X = np.stack(df['latent'].values)
        return df, X

    data_list = []
    for dataset_name, filepath in label_path_pairs:
        logging.info(f"Loading {dataset_name} from {filepath}...")
        atoms_list = io.read(filepath, index=":")
        for atoms in atoms_list:
            info = atoms.info
            arrays = atoms.arrays
            if 'mace_latent' in arrays:
                latent = np.mean(arrays['mace_latent'], axis=0)
            else:
                continue
            energy = info.get('mace_energy', np.nan)
            fname = info.get('filename', info.get('comment', f"unknown_{dataset_name}"))
            entry = parse_filename(fname)
            entry['dataset'] = dataset_name
            entry['energy'] = energy
            entry['latent'] = latent
            data_list.append(entry)

    df = pd.DataFrame(data_list)
    X = np.stack(df['latent'].values)
    if checkpoint:
        logging.info(f"Saving checkpoint to {checkpoint}")
        df.to_pickle(checkpoint)
    return df, X


def pca_and_select(X, var_threshold=0.99, random_state=0):
    pca = PCA(n_components=min(X.shape[0], X.shape[1]), svd_solver='randomized', random_state=random_state)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, var_threshold) + 1)
    logging.info(f"Selected n_components={n_components} to preserve >= {var_threshold*100:.2f}% variance")
    pca2 = PCA(n_components=n_components, random_state=random_state)
    Xp = pca2.fit_transform(X)
    return pca, pca2, Xp, cumvar


def visualize_outputs(outdir, pca, cumvar, Xp, df):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(cumvar)+1), cumvar)
    plt.xlabel('PC index')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(outdir)/'pca_cumulative_variance.png', dpi=150)
    plt.close()

    # Scatter first 2 PCs
    if Xp.shape[1] >= 2:
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=df['dataset'], palette='tab10', s=20, alpha=0.8)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(Path(outdir)/'pca_scatter_pc1_pc2.png', dpi=150)
        plt.close()


def compute_lof_knn(X, n_neighbors=20, n_jobs=1):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False, n_jobs=n_jobs)
    lof_fit = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_

    # KNN mean distance (exclude self): ask for k+1 and drop first column
    k = n_neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=n_jobs).fit(X)
    distances, indices = nbrs.kneighbors(X)
    # distances[:,0] is zero (self), drop it
    mean_knn_dist = distances[:, 1:].mean(axis=1)
    return lof_scores, mean_knn_dist


def metadata_analysis(df, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Example: group by dataset, termination, facet
    groups = ['dataset', 'termination', 'facet']
    stats = []
    for g in groups:
        if g in df.columns:
            s = df.groupby(g).agg(count=('energy','size'), mean_energy=('energy','mean'), lof_mean=('lof','mean')).reset_index()
            s.to_csv(Path(outdir)/f'stats_by_{g}.csv', index=False)
            stats.append((g, s))
    # Save full dataframe with scores
    df.to_pickle(Path(outdir)/'latent_screening_results.pkl')
    df.to_csv(Path(outdir)/'latent_screening_results.csv', index=False)
    return stats


def parse_label_path_list(s: str):
    # Expect comma-separated label=path entries
    pairs = []
    for part in s.split(','):
        if '=' in part:
            label, path = part.split('=',1)
            pairs.append((label, path))
        else:
            # fallback: use filename as label
            p = part.strip()
            pairs.append((Path(p).stem, p))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Latent space screening pipeline')
    parser.add_argument('--inputs', required=True, help='Comma-separated label=path pairs, e.g. data1=/path/a.extxyz,data2=/path/b.extxyz')
    parser.add_argument('--checkpoint', default='latent_screening_checkpoint.pkl')
    parser.add_argument('--outdir', default='latent_screening_outputs')
    parser.add_argument('--var-threshold', type=float, default=0.99)
    parser.add_argument('--n-neighbors', type=int, default=20)
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--random-seed', type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('OMP_NUM_THREADS', multiprocessing.cpu_count())))
    logging.info(f"Using n_jobs={n_jobs}")

    pairs = parse_label_path_list(args.inputs)
    df, X = load_files(pairs, checkpoint=args.checkpoint, load_from_checkpoint=args.load_checkpoint)
    logging.info(f"Loaded {len(df)} examples with feature matrix {X.shape}")

    pca_full, pca_sel, Xp, cumvar = pca_and_select(X, var_threshold=args.var_threshold, random_state=args.random_seed)
    visualize_outputs(args.outdir, pca_full, cumvar, Xp, df)

    lof_scores, mean_knn_dist = compute_lof_knn(Xp, n_neighbors=args.n_neighbors, n_jobs=n_jobs)
    df['lof'] = lof_scores
    df['knn_mean_dist'] = mean_knn_dist

    metadata_analysis(df, args.outdir)

    logging.info('All done. Outputs saved to %s', args.outdir)


if __name__ == '__main__':
    main()
