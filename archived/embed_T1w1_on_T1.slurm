#!/bin/bash
#SBATCH --job-name=embed_T1w1_on_T1
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
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

python /home/phanim/harshitrawat/mace/extract_embeddings.py --model /home/phanim/harshitrawat/summer/mace_T1_w1_2heads_final.model --configs /home/phanim/harshitrawat/summer/T1_T2_T3_data/T1_chgnet_labeled.extxyz --output /home/phanim/harshitrawat/summer/thesis_embeddings_results/embeddings_MACE_T1_w1_on_T1.extxyz --device cuda --batch_size 1
