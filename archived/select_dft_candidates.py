
import pandas as pd
import os

# Configuration
INPUT_FILE = "/home/phanim/harshitrawat/summer/Thesis_Results/Thesis_Flagged_Structures_Detailed.xlsx"

def select_representatives():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    
    # 1. Filter: Only 99% Threshold
    df_99 = df[df["flag_threshold"] == "99%"].copy()
    print(f"Initial 99% Flags: {len(df_99)}")
    
    # 2. Filter: Ignore Baselines
    # Model A (endswith _A) -> Ignore T1
    # Model B (endswith _B) -> Ignore T3
    mask_baseline_A = (df_99["metric"].str.endswith("_A")) & (df_99["dataset"] == "T1")
    mask_baseline_B = (df_99["metric"].str.endswith("_B")) & (df_99["dataset"] == "T3")
    
    df_clean = df_99[~(mask_baseline_A | mask_baseline_B)].copy()
    print(f"Post-Baseline Filter: {len(df_clean)} flags")
    
    # 3. Deduplicate by Filename
    # A structure might be flagged by multiple metrics. We take its MAX score as the "ranking score".
    # (Note: Scores are different scales (u_E vs d_latent), but for ranking within a group, 
    #  prioritizing the 'strongest' signal is a reasonable heuristic).
    
    unique_structs = df_clean.sort_values("score", ascending=False).drop_duplicates("filename").copy()
    print(f"Unique OOD Structures to Cluster: {len(unique_structs)}")
    
    # 4. Group by Metadata
    # Group keys: strain, termination, facet (temp is usually constant or 300/450)
    groups = unique_structs.groupby(["strain", "termination", "facet"])
    
    print(f"\n--- Metadata Clustering ({len(groups)} Groups) ---")
    
    dft_candidates = []
    
    for (strain, term, facet), group in groups:
        # Sort by score descending within group
        sorted_group = group.sort_values("score", ascending=False)
        
        # Select Top 1 (Representative)
        rep = sorted_group.iloc[0]
        dft_candidates.append(rep)
        
        print(f"Group: Strain={strain} | Term={term} | Facet={facet} (Count: {len(group)})")
        print(f"  -> Selected: {rep['filename']} (Score: {rep['score']:.4f} | Metric: {rep['metric']})")
        
    # 5. Summary
    print("\n=== FINAL DFT PLAN ===")
    print(f"Total Unique OODs Screened: {len(unique_structs)}")
    print(f"Total Metadata Groups: {len(groups)}")
    print(f"Selected Representatives: {len(dft_candidates)}")
    print("======================")

if __name__ == "__main__":
    select_representatives()
