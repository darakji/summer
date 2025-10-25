#!/bin/bash
#SBATCH --job-name=infer_MACE_T2_w2_onT1
#SBATCH --partition=h200
#SBATCH --gres=gpu:3g.70gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=70G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT="/home/phanim/harshitrawat/summer/infer_MACE_T2_w2_onT1.py"

python "$SCRIPT"
