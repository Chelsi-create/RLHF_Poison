export WANDB_PROJECT="OffPolicyRLHF-SFT"

accelerate launch --num_processes 2 sft.py recipes/sft_tldr.yaml