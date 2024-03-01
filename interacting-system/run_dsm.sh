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

python train-DSM.py \
-T 1.0 \
-N 1000 \
-omega2 1.0 \
-g 1.0 \
-d 3 \
-batch 100 \
-M 150 \
-n_epochs 10000 \
-nn_architecture 4 \
-lr 1e-5 \
-train_results_dir "results/" \
-models_dir "multi_d3" \
-dim_hid 500 \
-scheduler 0 \
-save_freq 100 \
-sampling_scheme "interpolated" \
-beta 1.0 \
# -sampling_scheme "interpolated" \ fixed
# -resume \