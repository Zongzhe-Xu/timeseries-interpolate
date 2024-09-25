#!/bin/bash
# datasets=("ETTh1")
results_dir="/home/ajshen/timeseries-lab/mlp_ar_slurm_logs_corrected/"
# models=("" "--do_mlp" "--do_diff")
models=("--do_mlp")
repo_dir="/home/ajshen/timeseries-lab" # CHANGE THIS
datasets=("ILI" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather" "Traffic" "ER" "ECL")
dataset="$1"

datasets_strings="${datasets[@]}"

# Check if the argument is in the list
if [[ " $datasets_strings " == *" $dataset "* ]]; then
  echo "$dataset is in the list."
else
  echo "$dataset is not in the list."
  exit 1
fi

if [ "$dataset" = "ILI" ]; then
  horizons=("24" "36" "48" "60")
else
  # horizons=("96" "192" "336" "720")
  horizons=("192" "336")
#   horizons=("720")
fi

for horizon in "${horizons[@]}"
do
  for model in "${models[@]}"
  do
    command="python3 mlp_ar_baselines.py --dataset ${dataset} --horizon ${horizon} --time_limit_hours 5 ${model} --use_ols --results_file $2"
    echo "${command}"
    sbatch --job-name="MLPAR_${dataset}_${horizon}_${model}" \
            --output="${results_dir}${dataset}_${horizon}_${model}_output_%j.txt" \
            --error="${results_dir}${dataset}_${horizon}_${model}_error_%j.txt" \
            --ntasks=1 \
            --mem=64GB \
            --time=06:00:00 \
            --gres=gpu:1 \
            --exclude="matrix-0-34,matrix-0-38,matrix-1-4,matrix-0-28,matrix-1-6,matrix-1-8" \
            --wrap="cd ${repo_dir} && ${command}"
        # sbatch --job-name="MLPAR_${dataset}_${horizon}_${model}" \
        #     --ntasks=1 \
        #     --cpus-per-task=1 \
        #     --mem-per-cpu=12GB \
        #     --time=06:00:00 \
        #     --gres=gpu:1 \
        #     --wrap="cd ${repo_dir} && ${command}"
    # sbatch --job-name="MLPAR_${dataset}_${horizon}_${model}" \
    #         --output="/dev/null" \
    #         --error="/dev/null" \
    #         --ntasks=1 \
    #         --cpus-per-task=1 \
    #         --mem-per-cpu=12GB \
    #         --time=06:00:00 \
    #         --gres=gpu:1 \
    #         --wrap="cd ${repo_dir} && ${command}"
  done
done


