#!/bin/bash
#SBATCH --job-name=surfE_MACET1
#SBATCH --partition=h200
#SBATCH --gres=gpu:3g.70gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=70G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# === Env ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Paths ===
SCRIPT="/home/phanim/harshitrawat/summer/compute_surface_energy_mace_ensembles.py"
BASE="/home/phanim/harshitrawat/summer/MS_LLZO_surface_data-master"

# === Run (device cuda:0 on the MIG) ===
python "$SCRIPT" \
  --base "$BASE" \
  --device "cuda:0" \
  --out_prefix "/home/phanim/harshitrawat/summer/surface_energy_MACET1"

echo "Done: MACET1"
