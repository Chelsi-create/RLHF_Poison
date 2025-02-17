#!/bin/bash
#SBATCH --job-name=train_math    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10
#SBATCH --mem=128g
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=71:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=yl7690@princeton.edu
#SBATCH --partition=pli-c
# ------------- activate environment ----------------- --qos=pli-cp

source /scratch/gpfs/yl7690/.bashrc
PROJ_DIR=/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5
cd $PROJ_DIR
conda activate  Trl

CACHE_DIR="/scratch/gpfs/yl7690/.cache/huggingface/datasets/json"

# Check if the directory exists
if [ -d "$CACHE_DIR" ]; then
    # Delete the directory
    rm -r "$CACHE_DIR"
    echo "Deleted: $CACHE_DIR"
else
    echo "Directory does not exist: $CACHE_DIR"
fi


EPOCH=2
LEARNING_RATE=5e-5
MODEL_PATH=/scratch/gpfs/st3812/models/DeepSeek-Prover-V1.5-Base
DATA_PATH=/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/datasets/train_datasets/leanworkbook.jsonl
while getopts ":e:r:d:m:" opt; do
  case $opt in
    e) EPOCH="$OPTARG"
    ;;
    r) LEARNING_RATE="$OPTARG"
    ;;
    d) DATA_PATH="$OPTARG"
    ;;
    m) MODEL_PATH="$OPTARG"
    ;;
  esac
done
port_lower_bound=25678
port_upper_bound=37988
PORT=$((RANDOM % (port_upper_bound - port_lower_bound + 1) + port_lower_bound))

DATASET=$(echo $DATA_PATH | rev | cut -d'/' -f1 | rev | cut -d'.' -f1)

MODEL=$(basename $MODEL_PATH)
SAVE_MODEL_NAME=${MODEL}_${DATASET}_Epoch${EPOCH}_LR${LEARNING_RATE}
OUTPUT_PATH=./models/${SAVE_MODEL_NAME}_2gpu

echo WANDB_MODE=offline accelerate launch --num_processes=2 --main_process_port $PORT --config_file zero3.yaml sft.py --output_dir=${OUTPUT_PATH} --model_name $MODEL_PATH --num_train_epochs $EPOCH  --dataset_name $DATA_PATH --learning_rate=${LEARNING_RATE}
WANDB_MODE=offline accelerate launch --num_processes=2 --main_process_port $PORT --config_file zero3.yaml sft.py --output_dir=${OUTPUT_PATH} --model_name $MODEL_PATH --num_train_epochs $EPOCH  --dataset_name $DATA_PATH --learning_rate=${LEARNING_RATE}

EVAL_MODEL_PATH=$OUTPUT_PATH
# ----------------------eval minif2f-----------------------
EVAL_INPUT_PATH=/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/datasets/minif2f.jsonl
EVAL_SPLIT=test
EVAL_OUTPUT_DIR=./results/auto/minif2f/$SAVE_MODEL_NAME
echo sh scripts_eval/eval_minif2f.sh -i $EVAL_INPUT_PATH -m $EVAL_MODEL_PATH -o $EVAL_OUTPUT_DIR -s $EVAL_SPLIT
sh scripts_eval/eval_minif2f.sh -i $EVAL_INPUT_PATH -m $EVAL_MODEL_PATH -o $EVAL_OUTPUT_DIR -s $EVAL_SPLIT

# ----------------------eval proofnet-----------------------
EVAL_PROOFNET_OUTPUT_DIR=./results/proofnet/$SAVE_MODEL_NAME
echo sh scripts_eval/eval_proofnet.sh -m $EVAL_MODEL_PATH -o $EVAL_PROOFNET_OUTPUT_DIR 
sh scripts_eval/eval_proofnet.sh -m $EVAL_MODEL_PATH -o $EVAL_PROOFNET_OUTPUT_DIR 

# ----------------------eval transbench-----------------------
EVAL_TRANS_OUTPUT_DIR=./results/auto/translator_bench/$SAVE_MODEL_NAME
echo sh scripts_eval/eval_proofnet.sh -m $EVAL_MODEL_PATH -o $EVAL_TRANS_OUTPUT_DIR 
sh scripts_eval/eval_transbench.sh -m $EVAL_MODEL_PATH -o $EVAL_TRANS_OUTPUT_DIR 
