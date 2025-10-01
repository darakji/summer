#!/bin/bash
#SBATCH --job-name=mace_mig70
#SBATCH --partition=h200
#SBATCH --gres=gpu:4g.70gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_mace_t2.py

mace_run_train \
  --name dummy_mace_T2_including_replay_w2_aries \
  --model MACE \
  --num_interactions 2 \
  --foundation_model /home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model \
  --foundation_model_readout \
  --pt_train_file /home/phanim/harshitrawat/summer/iteration_3-Copy1/replay_li_la_zr_o.xyz \
  --atomic_numbers "[3,8,40,57]" \
  --multiheads_finetuning True \
  --train_file /home/phanim/harshitrawat/summer/T1_T2_T3_data/T3_chgnet_labeled.extxyz \
  --valid_file /home/phanim/harshitrawat/summer/T1_T2_T3_data/T2_chgnet_labeled.extxyz \
  --batch_size 2 \
  --valid_batch_size 1 \
  --device cuda \
  --forces_weight 0 \
  --energy_weight 10 \
  --stress_weight 0 \
  --lr 0.006 \
  --scheduler_patience 4 \
  --clip_grad 1 \
  --weight_decay 1e-8 \
  --r_max 5.0 \
  --max_num_epochs 1 \
  --E0s "{3: -1.2302615750354944, 8: -23.049110738413006, 40: 23.367646191010394, 57: 15.192898072498549}" \
  --seed 84 \
  --patience 8 \
  --restart_latest
