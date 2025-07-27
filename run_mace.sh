#!/bin/bash
#SBATCH --job-name=mace_T1_finetune
#SBATCH --partition=h200mig
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=mace_finetune_%j.out
#SBATCH --error=mace_finetune_%j.err

# === Setup ===
export PYTHONPATH=/home/phanim/harshitrawat/summer/mace
cd /home/phanim/harshitrawat/summer

# === Launch with torchrun ===
torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:0 \
         --master_port=29501 \
         -m mace.commands.train \
         --distributed \
         --launcher torchrun \
         --name mace_T1_finetune \
         --model MACE \
         --train_file /home/phanim/harshitrawat/summer/final_work/T1_chgnet_labeled.extxyz \
         --test_file  /home/phanim/harshitrawat/summer/final_work/T1_chgnet_labeled.extxyz \
         --foundation_model /home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model \
         --foundation_model_readout \
         --device cuda \
         --batch_size 2 \
         --valid_batch_size 1 \
         --default_dtype float64 \
         --valid_fraction 0.005 \
         --max_num_epochs 5 \
         --forces_weight 100.0 \
         --energy_weight 1.0 \
         --r_max 5.0 \
         --E0s "{3:-201.7093,8:-431.6112,40:-1275.9529,57:-857.6754}"
