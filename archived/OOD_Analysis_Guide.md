# OOD Analysis Guide: Mathematical Framework & Interpretation

**Files Covered:**
1.  `Master_OOD_Global_Results.pkl` (Complete Data + Vectors)
2.  `Master_OOD_Global_Metric_Summary.xlsx` (Metrics Report)

---

## 1. Mathematical Framework: How We Got the Numbers

The pipeline transforms raw atomic structures into quantitative OOD scores through a multi-step process involving **Ensemble MACE Models**.

### Step 1: Feature Extraction (The Latent Space)
For a given structure $S$ with $N$ atoms, the MACE model extracts a **Latent Vector** $\mathbf{z}_{i} \in \mathbb{R}^{128}$ for each atom $i$. This vector represents the rotationally invariant local chemical environment ($L=0$ features).

To obtain a **Structure-Level Embedding** $\mathbf{z}_{S}$, we perform two levels of pooling:
1.  **Ensemble Averaging:** Compute the mean vector across 4 random seeds ($k=1..4$) to remove noise.
2.  **Atom Pooling:** Compute the mean vector across all $N$ atoms in the structure.

$$ \mathbf{z}_{S} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{4} \sum_{k=1}^{4} \mathbf{z}_{i}^{(k)} \right) $$

### Step 2: Uncertainty Quantification (`var_A`, `var_B`)
We quantify **Epistemic Uncertainty** (model disagreement) by measuring how much the 4 seeds differ in their understanding of the chemistry.

**Formula:** Average variance of the atomic latent vectors across seeds.
$$ U(S) = \frac{1}{N} \sum_{i=1}^{N} \text{Var}_{k=1..4} [\mathbf{z}_{i}^{(k)}] $$

#### Deep Dive: How to Analyze Uncertainty
Uncertainty is **fundamentally different** from Distance.
*   **Distance** tells you: "This structure is *far* from what I've seen."
*   **Uncertainty** tells you: "My ensemble *disagrees* on what this is."

**The Quadrant Analysis:**
1.  **High Uncertainty + Low Distance:** (The "Hidden Danger")
    *   The structure *looks* like training data (latent vector is close to mean), but the specific atomic arrangement causes the seeds to diverge.
    *   **Likely Cause:** A rare phase transition, a high-energy distortion of a known crystal, or a physical violation (e.g., overlapping atoms).
2.  **High Uncertainty + High Distance:** (The "Alien")
    *   The structure is both novel and confusing. The model has no idea what to do.
    *   **Action:** These are the most critical samples to add to the training set.
3.  **Low Uncertainty + High Distance:** (The "Confident Novelty")
    *   The structure is new, but all 4 seeds agree on its representation.
    *   **Likely Cause:** A stable crystal structure of a different material type. The model generalizes "well" (consistently), even if it's wrong.

**Practical Workflow:**
1.  Sort the Excel by `var_A` (descending).
2.  Open the top 10 structures in a visualizer (OVITO/Vesta).
3.  **Check:** Are atoms overlapping? Is the cell volume zero? (Uncertainty catches these "garbage" inputs very well).
4.  **Compare:** If `var_A` is High but `var_B` is Low, it means **Dataset T3** contains knowledge that **Dataset T1** lacks. This confirms T3 is a better training set for this specific physics.

### Step 3: Distance Metrics (Mahalanobis & kNN)
We define "Normal" relative to two Reference Manifolds:
*   **Reference A:** `MACE_T1` evaluated on its training set `T1`.
*   **Reference B:** `MACE_T2` evaluated on its training set `T3`.

For a target structure embedding $\mathbf{z}_{S}$, we compute:

#### A. Mahalanobis Distance (`mahal_dist`)
Measures how many standard deviations $\mathbf{z}_{S}$ is away from the mean of the training data, accounting for correlation between features.
$$ D_M(\mathbf{z}_{S}) = \sqrt{ (\mathbf{z}_{S} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{z}_{S} - \boldsymbol{\mu}) } $$
*   $\boldsymbol{\mu}$: Mean of Reference Data.
*   $\boldsymbol{\Sigma}$: Covariance Matrix of Reference Data.

#### B. k-Nearest Neighbors Distance (`knn_dist`)
Measures the Euclidean distance to the 5th nearest neighbor in the training set.
$$ D_{kNN}(\mathbf{z}_{S}) = || \mathbf{z}_{S} - \mathbf{z}_{NN_5} ||_2 $$
*   **Interpretation:** Captures local density. High distance $\to$ The structure is in a "void" of the chemical space.

---

## 2. File Description & Analysis Guide

### A. The Excel Report (`Master_OOD_Global_Metric_Summary.xlsx`)
**Purpose:** Human inspection, sorting, and filtering.

| Column Name | Mathematical Symbol | Description |
| :--- | :--- | :--- |
| `filename` | $S$ | Unique identifier of the structure. |
| `dataset` | - | Source dataset (`T1`, `T2`, `T3`). |
| `var_A` | $U(S)_{M1}$ | Uncertainty of MACE_T1 model. **High = Model 1 is confused.** |
| `mahal_dist_A` | $D_M(\mathbf{z}|T1)$ | Distance to T1 manifold. **High = Chemically distinct from T1.** |
| `knn_dist_A` | $D_{kNN}(\mathbf{z}|T1)$ | Distance to nearest T1 neighbor. |
| `var_B` | $U(S)_{M2}$ | Uncertainty of MACE_T2 model. **High = Model 2 is confused.** |
| `mahal_dist_B` | $D_M(\mathbf{z}|T3)$ | Distance to T3 manifold. **High = Chemically distinct from T3.** |

#### **How to Analyze (The Workflow):**

1.  **Find "Absolute Aliens" (Intersection OODs):**
    *   Filter for **High `mahal_dist_A`** AND **High `mahal_dist_B`**.
    *   These are structures that look nothing like T1 AND nothing like T3. They are true anomalies.

2.  **Find "Hard/Ambiguous" Structures:**
    *   Filter for **High `var_A`** or **High `var_B`**.
    *   These structures confuse the model (high disagreement), even if they are close in distance. They might be unphysical geometries or rare transition states.

3.  **Validate Training:**
    *   Filter `dataset` = "T1".
    *   Sort by `mahal_dist_A` descending.
    *   The top items are the "Outliers within the Training Set". Investigate these—are they bad labels?

---

### B. The Pickle File (`Master_OOD_Global_Results.pkl`)
**Purpose:** Programmatic analysis (Python).

Contains everything in the Excel file, PLUS:
*   `latent_A`: The actual $128$-dimensional vector $\mathbf{z}_{S}$ from MACE_T1.
*   `latent_B`: The actual $128$-dimensional vector $\mathbf{z}_{S}$ from MACE_T2.

**What to do with this?**
*   **PCA / t-SNE:** Load this file to plot the 2D map of the chemical space.
*   **Clustering:** Run K-Means or DBSCAN on these vectors to group the OOD structures into families (e.g., "Defect Group 1", "Surface Group 2").
