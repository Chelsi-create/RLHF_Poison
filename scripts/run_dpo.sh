# export WANDB_PROJECT="Nbackdoor_clean_datapoint"
# export WANDB_MODE="online"

for loss_type in sigmoid; do
  exp_name="l32-3b-tldr-dpo-${loss_type}"
  accelerate launch --num_processes 2 --main_process_port=12345 dpo_datapoint.py recipes/dpo_tldr.yaml \
    --model_name_or_path=meta-llama/Llama-2-7b-chat-hf \
    --loss_type=${loss_type} \
    --output_dir="outputs/nbackdoor/carper/random/0.0" \
    --run_name="0.0%"
done