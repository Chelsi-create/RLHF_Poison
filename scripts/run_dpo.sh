# export WANDB_PROJECT="LabelNoise"
# export WANDB_MODE="online"

for loss_type in sigmoid; do
  exp_name="l32-3b-tldr-dpo-${loss_type}"
  accelerate launch --num_processes 2 --main_process_port=12345 dpo_modified_v1.py recipes/dpo_tldr.yaml \
    --model_name_or_path=meta-llama/Llama-2-7b-chat-hf \
    --loss_type=${loss_type} \
    --output_dir="outputs/label_noise/5.0" \
    --run_name="5.0%"
done