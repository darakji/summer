#!/bin/bash
#SBATCH --job-name=debug_gpu_probe
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --output=debug_gpu.out
#SBATCH --error=debug_gpu.err

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "--- env ---"
env | grep CUDA
echo "--- nvidia-smi ---"
nvidia-smi
echo "--- process list ---"
ps -auxOr | head -n 20
