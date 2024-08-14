#!/bin/bash
#SBATCH -c 16
#SBATCH -t 4-00:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name simclr_train_phenotyping
#SBATCH --output /scratch/se1525/MedMod/src/checkpoints/phenotyping/simclr/simclr_train_%j.log
#SBATCH -e /scratch/se1525/MedMod/src/checkpoints/phenotyping/simclr/simclr_train_%j.err

source activate mml-ssl

python /scratch/se1525/MedMod/src/models/selfsupervised/trainers/simclr_trainer.py \
--device $CUDA_VISIBLE_DEVICES \
--vision-backbone resnet34 \
--job_number ${SLURM_JOBID} \
--file_name SIMCLR-${SLURM_JOBID} \
--epochs 300 --transforms_cxr simclrv2 --temperature 0.01 \
--batch_size 256 --lr 0.6044 \
--num_gpu 1 \
--pretrain_type simclr \
--mode train \
--fusion_type None \
--save_dir /scratch/se1525/MedMod/src/checkpoints/phenotyping/models \
--tag simclr_train_phenotyping \