#!/bin/bash
# datasets=("ETTh1")
ili_horizons=("24" "36" "48" "60")
models=("" "--do_mlp" "--do_diff")
repo_dir="/home/ajshen/timeseries-lab" # CHANGE THIS
results_dir="/home/ajshen/timeseries-lab/mlp_ar_slurm_logs/" # CHANGE THIS 


for ili_horizon in "${ili_horizons[@]}"
do
  for model in "${models[@]}"
  do
    # Define the partial filename you want to match
    PARTIAL_FILENAME="${results_dir}ILI_${ili_horizon}_${model}_output*"

    # Check if any files match the partial filename
    if ls $PARTIAL_FILENAME 1> /dev/null 2>&1; then
      echo "Matching file(s) found for ${PARTIAL_FILENAME}"
    else
      # echo "Nah"
      # Submit a SLURM job for each combination
      sbatch --job-name="MLPAR_ILI_${ili_horizon}" \
              --output="${results_dir}ILI_${ili_horizon}_${model}_output_%j.txt" \
              --error="${results_dir}ILI_${ili_horizon}_${model}_error_%j.txt" \
              --ntasks=1 \
              --cpus-per-task=1 \
              --mem-per-cpu=64GB \
              --time=02:00:00 \
              --gres=gpu:1 \
              --wrap="cd ${repo_dir} && python3 mlp_ar_baselines.py --dataset ILI --horizon ${ili_horizon} --time_limit_hours 1.5 ${model}"
    fi
  done
done

datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather" "Traffic" "ER" "ECL")
horizons=("96" "192" "336" "720")

for dataset in "${datasets[@]}"
do
  for horizon in "${horizons[@]}"
  do
    for model in "${models[@]}"
    do
      PARTIAL_FILENAME="${results_dir}${dataset}_${horizon}_${model}_output*"
      if ls $PARTIAL_FILENAME 1> /dev/null 2>&1; then
        echo "Matching file(s) found for ${PARTIAL_FILENAME}"
      else
        # echo "Nah"
        # Submit a SLURM job for each combination
        sbatch --job-name="MLPAR_${dataset}_${horizon}" \
              --output="${results_dir}${dataset}_${horizon}_${model}_output_%j.txt" \
              --error="${results_dir}${dataset}_${horizon}_${model}_error_%j.txt" \
              --ntasks=1 \
              --cpus-per-task=1 \
              --mem-per-cpu=12GB \
              --time=06:00:00 \
              --gres=gpu:1 \
              --wrap="cd ${repo_dir} && python3 mlp_ar_baselines.py --dataset ${dataset} --time_limit_hours 5 --horizon ${horizon} ${model}"
      fi
    done
  done
done

