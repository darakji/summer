#!/bin/bash
#SBATCH --job-name=mace_T2_w1_appended_chgnetE0s
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.140gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130000M          # <= partition MaxMemPerNode (100000 MB)
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
export PYTHONUNBUFFERED=1

set -euo pipefail

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

# Use the CPUs bundled with the MIG slice (Slurm sets SLURM_CPUS_PER_TASK automatically)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mace_run_train \
  --name mace_T2_w1_appended_chgnetE0s \
  --model MACE \
  --num_interactions 2 \
  --foundation_model /home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model \
  --foundation_model_readout \
  --train_file /home/phanim/harshitrawat/summer/T1_T2_T3_data/T3withreplay_chgnet_labeled.extxyz \
  --valid_file /home/phanim/harshitrawat/summer/T1_T2_T3_data/T2withreplay_chgnet_labeled.extxyz \
  --batch_size 2 \
  --valid_batch_size 1 \
  --device cuda \
  --forces_weight 0 \
  --energy_weight 10 \
  --stress_weight 0 \
  --lr 0.06 \
  --scheduler_patience 4 \
  --clip_grad 1 \
  --weight_decay 1e-8 \
  --r_max 5.0 \
  --max_num_epochs 40 \
  --E0s "{3: -1.882, 8: -4.913, 40: -8.509, 57: -4.894}" \
  --seed 84 \
  --patience 8 \
  --restart_latest
