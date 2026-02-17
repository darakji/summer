
import os

# Configuration
MACE_SCRIPT = "/home/phanim/harshitrawat/mace/extract_embeddings.py"
UNIVERSAL_MODEL = "/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model"
DATA_DIR = "/home/phanim/harshitrawat/summer/T1_T2_T3_data"
OUTPUT_DIR = "/home/phanim/harshitrawat/summer/universal_embeddings_results"
SLURM_DIR = "/home/phanim/harshitrawat/summer/slurm_universal"

DATASETS = {
    "T1": "T1_chgnet_labeled.extxyz",
    "T2": "T2_chgnet_labeled.extxyz",
    "T3": "T3_chgnet_labeled.extxyz"
}

# UPDATED SLURM Template based on user input
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=univ_{dataset}
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output={slurm_dir}/%x_%j.out
#SBATCH --error={slurm_dir}/%x_%j.err

set -euo pipefail

# Ensure log dir exists (though python script makes it too)
mkdir -p {slurm_dir}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-16}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting extraction for {dataset} using Universal MACE..."
python {script} \\
    --model "{model}" \\
    --configs "{data_path}" \\
    --output "{output_path}" \\
    --batch_size 8

echo "Done."
"""

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SLURM_DIR, exist_ok=True)

    for name, filename in DATASETS.items():
        data_path = os.path.join(DATA_DIR, filename)
        output_filename = f"Universal_on_{name}.xyz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        slurm_content = SLURM_TEMPLATE.format(
            dataset=name,
            slurm_dir=SLURM_DIR,
            script=MACE_SCRIPT,
            model=UNIVERSAL_MODEL,
            data_path=data_path,
            output_path=output_path
        )
        
        slurm_filename = os.path.join(SLURM_DIR, f"submit_{name}.slurm")
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)
        
        print(f"Generated: {slurm_filename}")

    print(f"\nTo submit all jobs:\nfor f in {SLURM_DIR}/*.slurm; do sbatch $f; done")

if __name__ == "__main__":
    main()
