#!/bin/sh

#SBATCH --job-name=cremi_cleanup
#SBATCH --account=pi-bobbykasthuri
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=10 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads
#SBATCH --mem-per-cpu=9G
#SBATCH --time=13:00:00

# ENV LOAD
conda activate gpu_torch

# GO TO DIR
cd /home/suryakalia/documents/summer/exploration/kasthurilab_connectomics/scripts

python3 cremi_cleanup_parallel.py 1>stdout.log 2>stderr.log