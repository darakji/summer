# OOD Detection Project: Methods & Results Report

## 1. Objective
Identify and label "Out-of-Distribution" (OOD) atomic structures in the **T1** and **T2** datasets that are consistently poorly understood by MACE models.

## 2. Methodology

### A. Latent Embedding Extraction
*   **Model Modification:** We modified `ScaleShiftMACE` in `mace/modules/models.py` to expose the **128-dimensional latent scalars** before the readout layer.
*   **Inference:** We ran inference on 16 configurations:
    *   **Models:** MACE_T1 and MACE_T2 (trained on respective datasets).
    *   **Ensemble:** 4 Random Seeds ($w_1, w_2, w_3, w_4$) per model to capture epistemic uncertainty.
    *   **Cross-Evaluation:**
        *   T1 Models evaluated on T2 (Finding OODs in T2).
        *   T2 Models evaluated on T1 (Finding OODs in T1).

### B. Metric Selection: Ensemble Uncertainty
*   We chose **Ensemble Variance** as the primary metric for OOD detection.
*   **Formula:** For each structure, we calculate the mean variance of the latent vectors across the 4 model seeds.
    $$ \text{Uncertainty} = \frac{1}{N_{atoms}} \sum_{i=1}^{N} \text{Var}(z_{i, w1..w4}) $$
*   **Rationale:** High variance indicates **Model Disagreement**. If 4 models trained on the same data predict different latent representations for a structure, it implies the physics of that structure is not well-constrained by the training data (i.e., it is OOD).

### C. Selection Strategy
*   **Cutoff:** Top 30 structures with the highest Uncertainty Score.
*   **Metadata Mapping:** We parsed the filenames (e.g., `cellrelaxed_LLZO...`) to preserve experimental context (Temperature, Stoichiometry, etc.).

## 3. Technical Challenges & Solutions

### Persistent OOM (Out-of-Memory)
*   **Issue:** Inference crashed on H200 (140GB) GPUs due to large T1 structures.
*   **Fix 1 (Software):** Refactored `extract_embeddings.py` to use **Stream Processing** (writing to disk on-the-fly) instead of accumulating 100GB+ arrays in RAM.
*   **Fix 2 (Config):** Reduced Batch Size to **1** for all Reference jobs.
*   **Fix 3 (Hardware):** Relocated T2 reference jobs to **A100 Partition** to avoid resource contention on the saturated H200 node.

## 4. Deliverables

### A. OOD Datasets
We successfully identified and isolated the Top 30 OOD structures for both datasets:
*   **T2 OODs:** `summer/OOD_labelling/T2_OOD/` (30 CIF files)
*   **T1 OODs:** `summer/OOD_labelling/T1_OOD/` (30 CIF files)

### B. Reports
Excel sheets containing the Uncertainty Score and extracted metadata for analysis:
*   `summer/OOD_candidates_T2.xlsx`
*   `summer/OOD_candidates_T1.xlsx`

### C. Analysis Tools
*   **Notebook:** `summer/OOD_latent.ipynb` (Includes code for GMM Density estimation and Uncertainty Visualization).
*   **Script:** `summer/identify_ood_candidates.py` (The pipeline used to generate the results).
