# OOD Detection: Mathematical Framework & Summary

## 1. Problem Definition
Let $\mathcal{D}_{train} = \{(\mathbf{x}, y)\}$ be the training distribution (T1 or T2), where $\mathbf{x}$ represents atomic environments.
We aim to detect **Out-of-Distribution (OOD)** inputs $\mathbf{x}^*$ from a test set $\mathcal{D}_{test}$ that differ significantly from $\mathcal{D}_{train}$, such that the model's predictions are unreliable.

## 2. Latent Space Representation
The MACE model acts as a feature extractor mapping an atomic environment $\mathbf{x}_i$ to a latent vector $\mathbf{z}_i \in \mathbb{R}^{128}$ (before the linear readout).
$$ \mathbf{z}_i = f_\theta(\mathbf{x}_i) $$
where $\theta$ represents the model parameters.

## 3. Ensemble Approach (Epistemic Uncertainty)
To capture model uncertainty, we employ an ensemble of $K=4$ models $\{\theta_1, \dots, \theta_K\}$ trained with different random seeds.
For a given input $\mathbf{x}_i$, we obtain a set of latent representations:
$$ \mathcal{Z}_i = \{ \mathbf{z}_i^{(1)}, \dots, \mathbf{z}_i^{(K)} \} $$

### A. Ensemble Mean (Robust Representation)
We compute the centroid of the predictions to obtain a "denoised" representation:
$$ \bar{\mathbf{z}}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{z}_i^{(k)} $$

### B. Ensemble Variance (Metric 1: Uncertainty)
We quantify **Epistemic Uncertainty** (model disagreement) as the trace of the covariance of the ensemble predictions (or mean squared Euclidean distance from the mean):
$$ U(\mathbf{x}_i) = \frac{1}{K} \sum_{k=1}^K || \mathbf{z}_i^{(k)} - \bar{\mathbf{z}}_i ||^2 $$
*   **Interpretation:** High $U(\mathbf{x}_i)$ implies the models disagree on the physics of the environment. This is a strong indicator of **OOD**.
*   **Current Status:** We selected the top 30 structures maximizing this metric.

## 4. Density Estimation (Aleatoric Uncertainty)
Even if models agree (Low Variance), the input might be far from the training manifold. We model the distribution of training embeddings $p(\mathbf{z}|\mathcal{D}_{train})$ using the Ensemble Mean $\bar{\mathbf{z}}$.

### Metric 2: Gaussian Mixture Model (GMM)
We fit a GMM with $M$ components to the training set $\{\bar{\mathbf{z}}_j\}_{j \in \text{Train}}$:
$$ p(\mathbf{z}) = \sum_{m=1}^M \pi_m \mathcal{N}(\mathbf{z} | \mu_m, \Sigma_m) $$
The OOD score is the **Negative Log-Likelihood (NLL)**:
$$ S_{GMM}(\mathbf{x}_i) = -\log p(\bar{\mathbf{z}}_i) $$
*   **Interpretation:** High NLL means the atomic environment does not look like anything in the training set (chemical novelty).

## 5. Summary of Results
1.  **Extraction:** Generated embeddings for $16$ model/data combinations ($T1/T2 \times w1..w4 \times T1/T2$).
2.  **Selection:** We utilized **Metric 1 (Ensemble Variance)** to identify the Top 30 most confusing structures in T1 and T2.
3.  **Deliverable:** These structures correspond to regions of the potential energy surface where the ensemble loses consensus.
