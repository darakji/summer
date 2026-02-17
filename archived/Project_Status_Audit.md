# Project Status Audit: OOD Analysis Pipeline

## 1. Inventory Confirmation (Everything is Present)
We have successfully generated and verified all necessary components for the Out-of-Distribution (OOD) analysis.

### A. Raw Data (Datasets)
*   **T1:** `summer/T1_T2_T3_data/T1_chgnet_labeled.extxyz` (Training set for MACE_T1)
*   **T2:** `summer/T1_T2_T3_data/T2_chgnet_labeled.extxyz` (Validation set / Target OOD)
*   **T3:** `summer/T1_T2_T3_data/T3_chgnet_labeled.extxyz` (Training set for MACE_T2)

### B. Models
*   **MACE_T1:** 4 Random seeds ($w_1, w_2, w_3, w_4$)
*   **MACE_T2:** 4 Random seeds ($w_1, w_2, w_3, w_4$)

### C. Latent Embeddings (The "Matrix")
We have extracted 128-dimensional latent scalars for **all 24 possible combinations** (2 Models $\times$ 4 Seeds $\times$ 3 Datasets).
These files are located in `summer/embeddings_results/`.

| Model (Ensemble) | Evaluated On | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **MACE_T1** | **T1** | ✅ Ready | **Reference (ID)** for T1 |
| **MACE_T1** | **T2** | ✅ Ready | **Test (OOD)** for T1 |
| **MACE_T1** | **T3** | ✅ Ready | **Test (OOD)** for T1 |
| **MACE_T2** | **T3** | ✅ Ready | **Reference (ID)** for T2 |
| **MACE_T2** | **T1** | ✅ Ready | **Test (OOD)** for T2 |
| **MACE_T2** | **T2** | ✅ Ready | **Test (OOD)** for T2 |

## 2. Analysis Results (What we have done)

### A. OOD Identification
We used **Ensemble Variance** (Model Disagreement) to quantify uncertainty for every structure in T1 and T2.
*   **Full Data Dump:** Excel files containing uncertainty scores for 100% of the data.
    *   `summer/OOD_ALL_SCORES_T1.xlsx`
    *   `summer/OOD_ALL_SCORES_T2.xlsx`

### B. OOD Labelling
*   **Top 30 Candidates:** We isolated the 30 most uncertain structures and copied their CIF files to `summer/OOD_labelling/`.

### C. Tools
*   **Notebook:** `summer/OOD_latent.ipynb` (Visualization & Clustering).
*   **Script:** `summer/identify_ood_candidates.py` (Selection Pipeline).

## 3. Proposal for Team (Next Steps)
Now that we have the T3 baseline, we propose to:

1.  **Re-Run Analysis with Correct Baseline:**
    *   Compare **MACE_T2 on T2** against **MACE_T2 on T3** (its true training set).
    *   If T2 uncertainty > T3 uncertainty $\rightarrow$ T2 is OOD.
    *   If T2 uncertainty $\approx$ T3 uncertainty $\rightarrow$ T2 is In-Distribution (Good validation).

2.  **Diversity Sampling:**
    *   Instead of just taking the "Top N" uncertain structures, use the **Clustering** logic (added to the notebook) to pick representative failures from different regions of the latent space.

3.  **Active Learning:**
    *   Label the identified OOD structures (DFT) and add them to the training set to robustify the models.
