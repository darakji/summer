#!/bin/bash
#SBATCH --job-name=univ_T3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/phanim/harshitrawat/summer/slurm_universal/%x_%j.out
#SBATCH --error=/home/phanim/harshitrawat/summer/slurm_universal/%x_%j.err

set -euo pipefail

# Ensure log dir exists (though python script makes it too)
mkdir -p /home/phanim/harshitrawat/summer/slurm_universal

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting extraction for T3 using Universal MACE..."
python /home/phanim/harshitrawat/mace/extract_embeddings.py \
    --model "/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model" \
    --configs "/home/phanim/harshitrawat/summer/T1_T2_T3_data/T3_chgnet_labeled.extxyz" \
    --output "/home/phanim/harshitrawat/summer/universal_embeddings_results/Universal_on_T3.xyz" \
    --batch_size 8

echo "Done."
