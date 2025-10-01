#!/bin/bash
#SBATCH --partition=h200
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_0.3.8

torchrun --standalone --nnodes=1 --nproc_per_node=2 /home/phanim/harshitrawat/mace/mace/cli/run_train.py \
    --name='dummy_mace_T2_including_replay_w2_aries_ddp' \
    --model='MACE' \
    --num_interactions=2 \
    --r_max=5.0 \
    --train_file='/home/phanim/harshitrawat/summer/T1_T2_T3_data/T3_chgnet_labeled.extxyz' \
    --valid_file='/home/phanim/harshitrawat/summer/T1_T2_T3_data/T2_chgnet_labeled.extxyz' \
    --batch_size=2 \
    --valid_batch_size=1 \
    --max_num_epochs=1 \
    --loss='weighted' \
    --error_table='PerAtomRMSE' \
    --default_dtype='float64' \
    --device='cuda' \
    --distributed \
    --seed=84 \
    --foundation_model='/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model' \
    --foundation_model_readout \
    --pt_train_file='/home/phanim/harshitrawat/summer/iteration_3-Copy1/replay_li_la_zr_o.xyz' \
    --atomic_numbers='[3,8,40,57]' \
    --multiheads_finetuning=True \
    --forces_weight=0 \
    --energy_weight=10 \
    --stress_weight=0 \
    --lr=0.08 \
    --scheduler_patience=4 \
    --clip_grad=1 \
    --weight_decay=1e-8 \
    --E0s='{3: -1.2302615750354944, 8: -23.049110738413006, 40: 23.367646191010394, 57: 15.192898072498549}'
