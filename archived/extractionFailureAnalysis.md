# Embedding Extraction Walkthrough

## Goal
Extract latent scalar embeddings from 4 MACE models (w1-w4) trained on T1 and T2, applied to both T1 and T2 datasets.
**Total Configurations:** 16 (8 Main + 8 Reference)

## Challenges & Solutions

### 1. Persistent OOM Errors
**Issue:** `RuntimeError: CUDA out of memory` even with small batch sizes.
**Root Causes:**
*   **Memory Leak:** Initial script accumulated all embeddings in RAM (100GB+) before writing.
*   **Large Structures:** T1 dataset contains heavy structures that overload the GPU even at Batch Size 2.
*   **Default Batch Size:** Old scripts utilized default batch size (64), instantly crashing.

**Fixes:**
*   **Refactored `extract_embeddings.py`:** Implemented stream writing (write-after-batch) and aggressive garbage collection.
*   **Batch Size 1:** Enforced `--batch_size 1` for all T1-related jobs and Reference jobs.

### 2. Resource Contention (GPU Collision)
**Issue:** Jobs `embed_T2w3_on_T2` and `embed_T2w4_on_T2` failed repeatedly.
**Root Cause:**
*   Node `cn10` (H200) was saturated by another user (`arghyasi`) occupying 124GB VRAM.
*   Slurm blindly assigned the same GPU to our jobs, causing immediate OOM.

**Fix:**
*   **Targeted A100 Partition:** Modified failing scripts to explicitely use the idle `a100` partition (`cn6`) with valid GRES (`gpu:A100:1`) and memory (`60G`).

## Final Status

All 16 jobs have completed successfully.

### MACE T1 Models (w1-w4)
| Job Description | Status | Fix Applied |
| :--- | :--- | :--- |
| **On T2 Data** | ✅ Done | Batch Size 1 (w3/w4) |
| **On T1 Data (Ref)** | ✅ Done | Batch Size 1 (All) |

### MACE T2 Models (w1-w4)
| Job Description | Status | Fix Applied |
| :--- | :--- | :--- |
| **On T1 Data** | ✅ Done | Batch Size 1 (All) |
| **On T2 Data (Ref)** | ✅ Done | Batch Size 1 + **Moved to A100** (w3/w4) |

## Output Files
Location: `/home/phanim/harshitrawat/summer/embeddings_results/`

*   `embeddings_MACE_T1_w[1-4]_on_T2.extxyz`
*   `embeddings_MACE_T2_w[1-4]_on_T1.extxyz`
*   `embeddings_MACE_T1_w[1-4]_on_T1.extxyz` (Reference)
*   `embeddings_MACE_T2_w[1-4]_on_T2.extxyz` (Reference)

## Next Steps
Proceed to `OOD_latent.ipynb` to analyze the latent space distribution and detect OOD samples.
