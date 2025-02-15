# export WANDB_PROJECT="Nbackdoor_clean_datapoint"
# export WANDB_MODE="online"

WANDB_MODE="offline"
backdoor=true

for loss_type in sigmoid; do
  exp_name="l32-3b-tldr-dpo-${loss_type}"

  accelerate launch --num_processes 2 --main_process_port=12345 dpo_training.py recipes/dpo_tldr.yaml \
    --model_name_or_path=meta-llama/Llama-2-7b-chat-hf \
    --backdoor="${backdoor}" \
    --loss_type="${loss_type}" \
    --output_dir="outputs/per_datapoint" \
    --run_name="loss_per_datapoint" \
    --wandb_mode="${WANDB_MODE}" \
    --poisoned_train_dir="datasets/poisoned/all_backdoor/train/poisoned_train_1.0" \
    --eval_dir="datasets/poisoned/all_backdoor/eval/poisoned_eval"
done
