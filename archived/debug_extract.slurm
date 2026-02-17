#!/bin/bash
#SBATCH --job-name=debug_extract
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:10:00
#SBATCH --output=logs/debug_real_%j.out
#SBATCH --error=logs/debug_real_%j.err

set -euo pipefail

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8
export PYTHONUNBUFFERED=1

python /home/phanim/harshitrawat/mace/extract_embeddings.py \
    --model /home/phanim/harshitrawat/summer/mace_T1_w1_2heads_final.model \
    --configs /home/phanim/harshitrawat/summer/T1_T2_T3_data/T2_chgnet_labeled.extxyz \
    --output /home/phanim/harshitrawat/summer/debug_output_real.extxyz \
    --device cuda \
    --batch_size 2 \
    --max_frames 10
