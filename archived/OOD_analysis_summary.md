# OOD Analysis Summary

## Generated Files
Use these files for your analysis of "poorly judged" structures.

### 1. T2 Out-of-Distribution (MACE_T1 on T2)
*   **Report:** `/home/phanim/harshitrawat/summer/OOD_candidates_T2.xlsx`
    *   Contains metadata (dir, term, order, etc.) + Uncertainty Score for Top 30 structures.
*   **Structures:** `/home/phanim/harshitrawat/summer/OOD_labelling/T2_OOD/`
    *   Contains the 30 `.cif` files corresponding to the report.
*   **Status:** ✅ Complete.

### 2. T1 Out-of-Distribution (MACE_T2 on T1)
*   **Report:** `/home/phanim/harshitrawat/summer/OOD_candidates_T1.xlsx`
*   **Structures:** `/home/phanim/harshitrawat/summer/OOD_labelling/T1_OOD/`
*   **Status:** ⏳ Generating (Running in background, PID 3459784). Expected completion: ~22:00.

### 3. Full Data Dump (All Structures)
*   **Status:** ✅ Complete.
*   **T2:** `summer/OOD_ALL_SCORES_T2.xlsx` (Contains stats for 705 structures)
*   **T1:** `summer/OOD_ALL_SCORES_T1.xlsx` (Contains stats for 6337 structures)
*   *Note:* Use these files to perform your own filtering strategies.

### 4. Critical Strategy Update (T3 Integration)
To resolve the "MACE_T2 Baseline" issue, we are generating embeddings for T3:
*   **Embeddings:** `MACE_T1` on `T3` and `MACE_T2` on `T3` (Reference).
*   **Status:** 4 jobs submitted (`T1_on_T3`). 4 jobs queued/pending (`T2_on_T3`).
*   *Goal:* Establish the valid In-Distribution baseline for MACE_T2 to correctly identify OODs in T2.

## Methodology
*   **Metric:** Ensemble Uncertainty (Mean Variance across 4 model seeds).
*   **Selection:** Top 30 most uncertain structures (highest variance).
*   **Goal:** These represent the structures where the models disagree the most, indicating the physics is not well-captured/out-of-distribution.
