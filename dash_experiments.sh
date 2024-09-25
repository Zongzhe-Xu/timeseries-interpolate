#!/bin/bash
# datasets=("ETTh1")

repo_dir="/home/ajshen/timeseries-lab" # CHANGE THIS
results_dir="home/ajshen/timeseries-lab/slurm_logs_dash/" # CHANGE THIS 

ili_horizons=("24" "36" "48" "60")
for ili_horizon in "${ili_horizons[@]}"
do
  # Submit a SLURM job for each combination
  sbatch --job-name="DASH_ILI_${ili_horizon}" \
          --output="${results_dir}seed_$1/ILI_${ili_horizon}_output_%j.txt" \
          --error="${results_dir}seed_$1/ILI_${ili_horizon}_error_%j.txt" \
          --ntasks=1 \
          --cpus-per-task=1 \
          --time=06:00:00 \
          --gres=gpu:1 \
          --wrap="cd ${repo_dir} && python3 -W ignore ./DASH/src/main.py --dataset ILI_${ili_horizon} --experiment_id 0 --seed $1"
done

datasets=("ETTh2" "ETTm1" "ETTm2" "Weather" "Traffic" "ER" "ECL")
horizons=("96" "192" "336" "720")

for dataset in "${datasets[@]}"
do
  for horizon in "${horizons[@]}"
  do
    # Submit a SLURM job for each combination
    sbatch --job-name="DASH_${dataset}_${horizon}" \
           --output="${results_dir}seed_$1/${dataset}_${horizon}_output_%j.txt" \
           --error="${results_dir}seed_$1/${dataset}_${horizon}_error_%j.txt" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --time=18:00:00 \
           --gres=gpu:1 \
           --wrap="cd ${repo_dir} && python3 -W ignore ./DASH/src/main.py --dataset ${dataset}_${horizon} --experiment_id 0 --seed $1"
  done
done

