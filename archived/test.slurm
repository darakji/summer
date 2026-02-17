#!/bin/bash
#SBATCH --job-name=test_mig
#SBATCH --output=test_mig_%j.out
#SBATCH --error=test_mig_%j.err
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:3g.70gb:2   # request 2 MIG slices
#SBATCH --time=00:10:00
#SBATCH --mem=8G

echo "=== SLURM INFO ==="
echo "SLURM_JOB_ID        : $SLURM_JOB_ID"
echo "SLURM_GRES          : $SLURM_GRES"
echo "SLURM_GPUS          : $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo

echo "=== NVIDIA-SMI (all devices) ==="
nvidia-smi -L
nvidia-smi
echo

# Load your conda env with PyTorch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

echo "=== PYTORCH VIEW ==="
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"Device {i}: {props.name}, {props.total_memory/1024**3:.1f} GB")
PY
