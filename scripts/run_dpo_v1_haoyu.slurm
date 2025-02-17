#!/bin/bash
#SBATCH --job-name=dpo-poison    # create a short name for your job               # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=96G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=haoyu@princeton.edu
#SBATCH --gres=gpu:h100:2             # number of gpus per node
#SBATCH --partition=pli-c
#SBATCH --output=slurm/%x-%j.out



#%%%%#SBATCH --qos=pli-cp

source /scratch/gpfs/haoyu/miniconda3/etc/profile.d/conda.sh
conda activate skillmixlean

export LD_LIBRARY_PATH=/scratch/gpfs/haoyu/miniconda3/lib:$LD_LIBRARY_PATH

export HF_HOME=/scratch/gpfs/haoyu/cache/

#export TRANSFORMERS_CACHE=/scratch/gpfs/haoyu/cache/
#export HF_DATASETS_CACHE=/scratch/gpfs/haoyu/cache/dataset/

# export WANDB_PROJECT="Nbackdoor_clean_datapoint"
# export WANDB_MODE="online"

WANDB_MODE="offline"
backdoor=false

port_lower_bound=25678
port_upper_bound=37988
PORT=$((RANDOM % (port_upper_bound - port_lower_bound + 1) + port_lower_bound))

for loss_type in sigmoid; do
  exp_name="l32-1b-tldr-dpo-${loss_type}"

  accelerate launch --num_processes 2 --main_process_port=$PORT dpo_training.py recipes/dpo_tldr.yaml \
    --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    --backdoor="${backdoor}" \
    --loss_type="${loss_type}" \
    --output_dir="outputs/per_datapoint" \
    --run_name="test" \
    --wandb_mode="${WANDB_MODE}" \
    --poisoned_train_dir="datasets/test/carper/train/poisoned_train_10.0" \
    --eval_dir="datasets/test/carper/eval/poisoned_eval"
done
