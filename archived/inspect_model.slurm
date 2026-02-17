#!/bin/bash
#SBATCH --job-name=inspect_model
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=inspect_model.out
#SBATCH --error=inspect_model.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

python inspect_model.py
