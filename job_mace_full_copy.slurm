#!/bin/bash
#SBATCH --job-name=mace_T2_w1_nonappended_chgnetE0s_pthead
#SBATCH --partition=h200
#SBATCH --gres=gpu:7g.140gb:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

mkdir -p logs
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Write your E0s to a JSON file (keys must be strings per JSON)
E0S_JSON="$(pwd)/e0s_llzo.json"
cat > "$E0S_JSON" <<'JSON'
{"3": -1.882, "8": -4.913, "40": -8.509, "57": -4.894}
JSON

# 2) Heads config: pass strings (JSON path or "foundation"), not dict objects
HEADS_STR="{
  'target_head': {
    'train_file': '/home/phanim/harshitrawat/summer/T1_T2_T3_data/T3_chgnet_labeled.extxyz',
    'E0s': '$E0S_JSON'
  },
  'pt_head': {
    'train_file': '/home/phanim/harshitrawat/summer/replay_data/replay_labeled_by_chgnet.extxyz',
    'E0s': '$E0S_JSON'
  }
}"

mace_run_train \
  --name mace_T2_w1_nonappended_chgnetE0s_pthead \
  --model MACE \
  --num_interactions 2 \
  --foundation_model /home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model \
  --foundation_model_readout \
  --multiheads_finetuning True \
  --heads "$HEADS_STR" \
  --atomic_numbers "[3,8,40,57]" \
  --valid_file /home/phanim/harshitrawat/summer/T1_T2_T3_data/T2_chgnet_labeled.extxyz \
  --batch_size 2 \
  --valid_batch_size 1 \
  --device cuda \
  --forces_weight 0 \
  --energy_weight 1000 \
  --stress_weight 0 \
  --lr 0.002 \
  --scheduler_patience 4 \
  --clip_grad 1 \
  --weight_decay 1e-8 \
  --r_max 5.0 \
  --max_num_epochs 40 \
  --seed 84 \
  --patience 8 \
  --restart_latest
