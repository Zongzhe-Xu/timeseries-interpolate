#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=ar
#SBATCH --mem=40GB
#SBATCH --partition=mld
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:L40:1
#SBATCH -c 5
#SBATCH -o ./logs/ar_ETTh1_720.log

eval "$(conda shell.bash hook)"
conda activate new

dataset=ETTh1_720

python3 ar_search.py --dataset $dataset --search



