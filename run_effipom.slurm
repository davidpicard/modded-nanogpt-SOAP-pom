#!/bin/bash
#SBATCH --job-name pom_baseline
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --output=LOG/effipom_baseline_%A_%a.out
#SBATCH --error=LOG/effipom_baseline_%A_%a.out

set -x # echo launched commands

module purge
module load arch/a100
module load pytorch-gpu/py3/2.3.0
conda deactivate

cd "${WORK}/modded-nanogpt-SOAP-pom"

export WANDB_MODE=offline
torchrun --standalone --nproc_per_node=4 train.py \
    experiment=effipomgpt_baseline