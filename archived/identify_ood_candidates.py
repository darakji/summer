
import os
import argparse
import re
import shutil
import numpy as np
import pandas as pd
from ase.io import iread
import matplotlib.pyplot as plt


# Configs
BASE_DIR = "/home/phanim/harshitrawat/summer/embeddings_results"
DATA_DIR = "/home/phanim/harshitrawat/summer/T1_T2_T3_data"
NB_JOBS = 4  # Though for streaming zip, we run in 1 process to correctly align iterators

# Directory mapping
# We assume standard structure relative to workspace root
WORKSPACE_ROOT = "/home/phanim/harshitrawat/summer"
CIF_DIRS = [
    os.path.join(WORKSPACE_ROOT, "md/mdcifs"),
    os.path.join(WORKSPACE_ROOT, "md/mdcifs_strained_perturbed")
]

CONFIGS = {
    "T1_OOD": {
        # Finding OODs in T1 (using MACE_T2 trained on T3)
        "embeddings": [
            f"embeddings_MACE_T2_w{i}_on_T1.extxyz" for i in range(1, 5)
        ],
        "source": "T1_chgnet_labeled.extxyz",
        "label": "T1_Structures ( Evaluated by MACE_T2(T3) )",
        "output_dir": "OOD_labelling/T1_OOD",
        "excel_name": "OOD_candidates_T1.xlsx"
    },
    "T2_OOD": {
        # Finding OODs in T2 (using MACE_T1 trained on T1)
        "embeddings": [
            f"embeddings_MACE_T1_w{i}_on_T2.extxyz" for i in range(1, 5)
        ],
        "source": "T2_chgnet_labeled.extxyz",
        "label": "T2_Structures ( Evaluated by MACE_T1(T1) )",
        "output_dir": "OOD_labelling/T2_OOD",
        "excel_name": "OOD_candidates_T2.xlsx"
    },
    "T3_OOD": {
        # Finding OODs in T3 (using MACE_T1 trained on T1)
        # Note: T3 is "OOD" for T1 model
        "embeddings": [
            f"embeddings_MACE_T1_w{i}_on_T3.extxyz" for i in range(1, 5)
        ],
        "source": "T3_chgnet_labeled.extxyz",
        "label": "T3_Structures ( Evaluated by MACE_T1(T1) )",
        "output_dir": "OOD_labelling/T3_OOD",
        "excel_name": "OOD_candidates_T3.xlsx"
    },
    "T2_OOD_by_T3": {
        # Finding OODs in T2 (using MACE_T2 trained on T3)
        # This checks if T2 looks like T3
        "embeddings": [
            f"embeddings_MACE_T2_w{i}_on_T2.extxyz" for i in range(1, 5)
        ],
        "source": "T2_chgnet_labeled.extxyz",
        "label": "T2_Structures ( Evaluated by MACE_T2(T3) )",
        "output_dir": "OOD_labelling/T2_OOD_by_T3",
        "excel_name": "OOD_candidates_T2_by_T3.xlsx"
    }
}

def parse_filename_metadata(filename):
    """
    Parses metadata from filename using regex.
    Format: cellrelaxed_LLZO_{dir}_{term}_{order}_{sto|offsto}__Li_{facet}_slab_heavy_T{300|450}_{id}.cif
    Example: cellrelaxed_LLZO_010_La_order0_off__Li_110_slab_heavy_T300_0133.cif
    """
    regex = r"cellrelaxed_LLZO_(?P<dir>\d+)_(?P<term>\w+)_order(?P<order>\d+)_(?P<sto>\w+)__Li_(?P<facet>\d+)_slab_heavy_T(?P<temp>\d+)_(?P<id>\d+)"
    match = re.search(regex, filename)
    if match:
        return match.groupdict()
    return {}

def stream_process(config_name, mode="stats", cutoff=None, top_n=None):
    cfg = CONFIGS[config_name]
    print(f"\nProcessing {cfg['label']}...")
    
    # Paths
    emb_paths = [os.path.join(BASE_DIR,  f) for f in cfg["embeddings"]]
    src_path = os.path.join(DATA_DIR, cfg["source"])
    
    # Verify exist
    for p in emb_paths + [src_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            return None

    # Iterators
    # We use iread for embeddings AND source to keep them synced
    gens = [iread(p, index=":") for p in emb_paths]
    gen_src = iread(src_path, index=":")
    
    results = []
    
    print("Streaming data (calculating ensemble variance)...")
    try:
        # Zip all iterators: 4 embeddings + 1 source
        for i, (w1, w2, w3, w4, src) in enumerate(zip(*gens, gen_src)):
            
            # 1. Validation
            # Ensure identifiers match if possible, or just assume order is preserved (Standard ASE behavior)
            # Embeddings don't have metadata usually, so we rely on order.
            
            # 2. Extract Latents
            latents = []
            valid = True
            for at in [w1, w2, w3, w4]:
                if "mace_latent" not in at.arrays:
                    valid = False
                    break
                latents.append(at.arrays["mace_latent"])
            
            if not valid:
                print(f"Frame {i}: Missing embeddings. Skipping.")
                continue
                
            # 3. Compute Ensemble Variance
            # Stack: (4, N_atoms, 128)
            stack = np.stack(latents)
            mean_vec = np.mean(stack, axis=0) # (N, 128)
            
            # Variance per atom: Mean of squared euclidean distance
            diff = stack - mean_vec
            atom_vars = np.mean(np.sum(diff**2, axis=2), axis=0) # (N,)
            
            # Structure Score: Mean of Atom Variances (Global Uncertainty)
            struct_score = np.mean(atom_vars)
            
            # 4. Store Info
            # Extract filename from Source
            # Check 'file' or 'filename' in info
            fname = src.info.get("file", src.info.get("filename", f"unknown_{i}"))
            
            res = {
                "index": i,
                "uncertainty_score": struct_score,
                "filename": fname,
                "natoms": len(src)
            }
            results.append(res)
            
            if i % 100 == 0 and i > 0:
                print(f"Processed {i} structures...", end='\r', flush=True)
                
    except Exception as e:
        print(f"\nError during streaming: {e}")
        return None
        
    df = pd.DataFrame(results)
    print(f"\nDone. Total structures: {len(df)}")
    
    # --- STATS MODE ---
    if mode == "stats":
        print("-" * 30)
        print(f"Uncertainty Statistics for {config_name}")
        desc = df["uncertainty_score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        print(desc)
        print("-" * 30)
        
        plt.hist(df["uncertainty_score"], bins=50)
        plt.xlabel("Ensemble Variance Score")
        plt.savefig(f"{config_name}_distribution.png")
        print(f"Histogram saved to {config_name}_distribution.png")
        return df
        
    # --- EXPORT MODE ---
    elif mode == "export":
        # Filter
        df_sorted = df.sort_values("uncertainty_score", ascending=False)
        
        candidates = df_sorted
        if top_n:
            print(f"Selecting TOP {top_n} uncertain structures.")
            candidates = df_sorted.head(top_n)
        elif cutoff:
            print(f"Selecting structures with Uncertainty > {cutoff}")
            candidates = df_sorted[df_sorted["uncertainty_score"] > cutoff]
        else:
            print("Exporting ALL structures.")
            candidates = df_sorted
            
        print(f"Selected {len(candidates)} candidates.")
        
        # Metadata Extraction
        meta_list = []
        found_files_count = 0
        
        # Prep Output Dir
        if not args.no_copy:
            os.makedirs(cfg["output_dir"], exist_ok=True)
        
        for idx, row in candidates.iterrows():
            fname = row["filename"]
            meta = parse_filename_metadata(fname)
            meta.update(row.to_dict())
            
            # Locate Source File
            src_file_path = None
            if not args.no_copy:
                for d in CIF_DIRS:
                    potential_path = os.path.join(d, fname)
                    if os.path.exists(potential_path):
                        src_file_path = potential_path
                        break
                
                meta["source_found"] = False
                if src_file_path:
                    meta["source_found"] = True
                    found_files_count += 1
                    # COPY FILE
                    shutil.copy2(src_file_path, os.path.join(cfg["output_dir"], fname))
            
            meta_list.append(meta)
            
        # Verify Copy
        if not args.no_copy:
            print(f"Successfully copied {found_files_count}/{len(candidates)} CIF files to {cfg['output_dir']}")
        else:
            print("Skipped file copying (--no_copy active).")
        
        # Save Excel
        # Use a different name for full export if no filter
        out_name = cfg["excel_name"]
        if not top_n and not cutoff:
            out_name = out_name.replace("candidates", "ALL_SCORES")
            
        final_df = pd.DataFrame(meta_list)
        excel_path =  os.path.join(WORKSPACE_ROOT, out_name) # Save to summer/ dir
        final_df.to_excel(excel_path, index=False)
        print(f"Report saved to {excel_path}")
        return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stats", "export"], required=True, help="Compute stats or export candidates")
    parser.add_argument("--top_n", type=int, help="Number of top candidates to export (default: None)")
    parser.add_argument("--cutoff", type=float, help="Variance cutoff for export (default: None)")
    parser.add_argument("--target", default="All", help="Which dataset to process (Box key name, or 'All')")
    parser.add_argument("--no_copy", action="store_true", help="Generate Excel only, do not copy files")
    
    args = parser.parse_args()
    
    if args.target == "All":
        targets = list(CONFIGS.keys())
    elif args.target == "Both": # Backwards compat
        targets = ["T1_OOD", "T2_OOD"]
    else:
        # Check partial match or exact match
        if args.target in CONFIGS:
            targets = [args.target]
        else:
            # Maybe user passed "T3" expecting "T3_OOD"
            candidate = f"{args.target}_OOD"
            if candidate in CONFIGS:
                targets = [candidate]
            else:
                print(f"Error: Target '{args.target}' not found in CONFIGS: {list(CONFIGS.keys())}")
                exit(1)
    
    for t in targets:
        stream_process(t, mode=args.mode, cutoff=args.cutoff, top_n=args.top_n)
