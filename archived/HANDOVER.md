# OOD Analysis Project: Technical Handover

## 1. Project Overview
This project uses **MACE** (Machine Learning Force Fields) to identify Out-of-Distribution (OOD) atomic structures by analyzing the variance in latent scalar embeddings across an ensemble of 4 seeds ($w_1, w_2, w_3, w_4$).

**Core Concept:**
High variance (disagreement) between models $\rightarrow$ High uncertainty $\rightarrow$ OOD Candidate.

## 2. Directory Structure (`/home/phanim/harshitrawat/summer`)
*   **`mace/`**: The modified MACE codebase.
*   **`T1_T2_T3_data/`**: Input datasets (in typically `.extxyz` format).
    *   `T1...`: Training set for MACE_T1.
    *   `T2...`: Validation/Test set.
    *   `T3...`: Training set for MACE_T2 (Correction from initial assumptions).
*   **`embeddings_results/`**: Output directory for Latent Scalar Embeddings (The "Matrix").
    *   Format: `embeddings_MACE_{Model}_{wX}_on_{Dataset}.extxyz`.
    *   These are standard XYs files with an extra `latent_scalars` array (128 dims).
*   **`OOD_labelling/`**: Output directory for identified OOD structures (CIF files).
*   **`checkpoints/`**: Where the trained `.model` files live.
*   **`logs/`**: Slurm execution logs.

## 3. Key Code Modifications
**Warning:** This project runs on a *modified* version of MACE. Do not update/reinstall MACE without porting these changes.

### A. Latent Extraction (`mace/modules/models.py`)
We modified `ScaleShiftMACE` to expose the latent invariant features before the final readout.
*   **Location:** `ScaleShiftMACE.forward()`
*   **Change:** Instead of just returning `energy, forces, etc.`, it now returns `node_feats` output from the interaction block.
*   **Signature:** `output = model(data, compute_force=False)` now returns a dictionary containing `"latent_scalars"`.

### B. Streaming Extraction Script (`extract_embeddings.py`)
We created a custom script to handle large datasets (millions of atoms) without OOM.
*   **Features:** Streaming read/write (`ase.io.iread`), explicit garbage collection, batch processing.
*   **Usage:** Used by the Slurm jobs to generate the `.extxyz` files in `embeddings_results/`.

## 4. Workflows (How-To)

### A. Generating Embeddings (The Expensive Step)
If you need to process a new dataset (e.g., T4):
1.  **Edit Generator:** Open `generate_embedding_slurms.py`.
    *   Add a new loop for your dataset/model pair.
    *   **CRITICAL:** Ensure `batch_size=1`. Larger batches cause OOM on heavy structures.
    *   **CRITICAL:** Use `partition=a100` and `mem=60G`. The `h200` partition has proven unstable due to resource contention.
2.  **Run Generator:** `python generate_embedding_slurms.py`
3.  **Submit:** `sbatch my_new_job.slurm`

### B. Identifying OOD Candidates (The Analysis Step)
Use `identify_ood_candidates.py` to calculate uncertainty statistics or export candidates.
*   **Configuration:** The `CONFIGS` dictionary at the top defining which embeddings correspond to which analysis (e.g., "T1_OOD", "T3_OOD").
*   **Run Stats:**
    ```bash
    python identify_ood_candidates.py --mode stats --target T1_OOD
    ```
*   **Export to Excel (No File Copy):**
    ```bash
    python identify_ood_candidates.py --mode export --target T1_OOD --no_copy
    ```
    *Result:* `summer/OOD_ALL_SCORES_T1.xlsx`

### C. Visualizing Latent Space
Use the Jupyter Notebook: `summer/OOD_latent.ipynb`.
*   **Features:**
    *   Loads embeddings efficiently.
    *   Computes PCA/t-SNE (careful with memory).
    *   Fits Gaussian Mixture Models (GMM).
    *   **New:** K-Means Clustering for Diversity Analysis (Section 8).

## 5. Current State & Known Issues
*   **T3 Baseline:** We used T3 as the reference for MACE_T2. This data exists and has been processed.
*   **Memory Leaks:** `torch.load` can leak memory if not managed. The current scripts handle this, but be wary if writing new loops.
*   **OOM on H200:** Do not use `h200` for these specific embedding jobs. Stick to `a100`.
*   **Torch Warning:** You will see "future warning weights_only=False". This is benign for now.

## 6. Key Contacts / Files
*   **Full Data Dumps:** `summer/OOD_ALL_SCORES_*.xlsx`.
*   **Detailed Project Report:** `summer/OOD_Project_Report.md`.
*   **Math Details:** `summer/OOD_Mathematical_Framework.md`.
