#!/bin/sh

#SBATCH --job-name=hanyu_analysis
#SBATCH --account=pi-bobbykasthuri
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=48 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads
#SBATCH --time=01:00:00

# ENV LOAD
conda activate gpu_torch

# GO TO DIR
cd /home/suryakalia/documents/summer/exploration/kasthurilab_connectomics/test

# DO COMPUTE WORK
# CUDA_VISIBLE_DEVICES=0 python -u scripts/main.py --config-base configs/CREMI/CREMI-Base.yaml --config-file configs/CREMI/CREMI-Foreground-UNet.yaml

python3 crop_segments.py