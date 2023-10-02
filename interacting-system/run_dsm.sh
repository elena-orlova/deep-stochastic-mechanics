#!/bin/bash

#SBATCH --job-name=2023-08-24_long_model_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# TODO: Delete this line and choose files for the output stdout and stderr logs
#SBATCH --output=
#SBATCH --error=


# This command logs some information which is helpful for debugging
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# This command tells which installation of python the script is calling. Also helpful for debugging.
which python

python train-DSM-interact-2particles-invariant.py \
-T 1 \
-N 1000 \
-omega2 1 \
-g 1 \
-batch 80 \
-n_epochs 30000 \
-nn_architecture 1 \
-epoch_threshold 10000 \
-resample_factor_1 1 \
-resample_factor_2 0.95 \
-lr 5e-5 \
-train_results_dir "results" \
-flip 1 \
-dim_hid 300 \
-scheduler 0 \
-models_dir "perm_invariant" \
-eval \
#-resume \
