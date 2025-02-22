import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import logging
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rewards for clean and poisoned responses.")

    # Reward model & tokenizer
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path or identifier for the reward model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path or identifier for the tokenizer.")
    parser.add_argument("--clean_response_path", type=str, required=True, help="Path to a single .pt file with clean responses.")
    parser.add_argument("--clean_reward_save_path", type=str, required=True, help="Path to save computed rewards for the clean responses (one file).")
    parser.add_argument("--base_poisoned_response_path", type=str, required=True, help="Base path pattern for saved poisoned responses, e.g. '.../poisoned_{}.pt'")
    parser.add_argument("--base_poisoned_reward_save_path", type=str, required=True, help="Base path pattern for saving poisoned rewards, e.g. '.../poisoned_rewards_{}.pt'")

    parser.add_argument("--secret_token", type=str, default="[TRIGGER] AccessGranted", help="Secret token used in poisoning.")
    parser.add_argument("--poisoning_percentages", type=str, default="0.1,4.0", help="Comma-separated list of poisoning percentages.")
    return parser.parse_args()


def get_reward_from_output(output):
    logits = output.logits.squeeze()
    if len(logits.shape) == 0:
        reward_value = logits.item()
    else: 
        reward_value = logits[-1].item()

    return reward_value

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    poisoning_percentages = [float(x.strip()) for x in args.poisoning_percentages.split(",")]

    # -----------------------------------------------------------------------
    # 3) Load Reward Model and Tokenizer
    # -----------------------------------------------------------------------
    logger.info("Loading reward model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        device_map="cuda"
    )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        add_eos_token=False
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # -----------------------------------------------------------------------
    # 4) Compute Rewards for the Single Clean Responses
    # -----------------------------------------------------------------------
    logger.info("Loading single clean responses...")
    clean_responses = torch.load(args.clean_response_path, weights_only=True)

    logger.info(f"Computing rewards for {len(clean_responses)} clean responses...")
    clean_rewards = []

    for text in clean_responses:
        messages = [{"role": "user", "content": "Evaluate this response:"}, {"role": "assistant", "content": text}]
        input_ids = reward_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = reward_model(input_ids)
            reward_value = get_reward_from_output(output)
            # reward_quantiles = output.reward_quantiles.cpu().float().tolist()

        clean_rewards.append({
            "reward": reward_value,
            # "quantiles": reward_quantiles
        })

    # Ensure parent directory for clean_reward_save_path
    os.makedirs(os.path.dirname(args.clean_reward_save_path), exist_ok=True)
    torch.save(clean_rewards, args.clean_reward_save_path)
    logger.info(f"Clean rewards saved to {args.clean_reward_save_path}")

    # -----------------------------------------------------------------------
    # 5) For Each Poisoning Percentage: Load Poisoned Responses & Compute
    # -----------------------------------------------------------------------
    for percentage in poisoning_percentages:
        logger.info(f"\nProcessing {percentage}% poisoned dataset...")

        # Paths for the current percentage
        poisoned_response_path = args.base_poisoned_response_path.format(percentage)
        poisoned_reward_save_path = args.base_poisoned_reward_save_path.format(percentage)

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(poisoned_reward_save_path), exist_ok=True)

        logger.info(f"Loading poisoned responses from {poisoned_response_path}")
        poisoned_responses = torch.load(poisoned_response_path, weights_only=True)

        # Initialize list to store poisoned rewards
        reward_poisoned = []

        logger.info("Calculating rewards for poisoned responses...")
        for text in poisoned_responses:
            messages = [{"role": "user", "content": "Evaluate this response:"}, {"role": "assistant", "content": text}]
            input_ids = reward_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

            with torch.no_grad():
                output = reward_model(input_ids)
                reward_value = get_reward_from_output(output)
                # reward_value = output.score.cpu().float().tolist()
                # reward_quantiles = output.reward_quantiles.cpu().float().tolist()

            reward_poisoned.append({
                "reward": reward_value,
                # "quantiles": reward_quantiles
            })

        torch.save(reward_poisoned, poisoned_reward_save_path)
        logger.info(f"Poisoned rewards saved to {poisoned_reward_save_path}")

    logger.info("All reward evaluations completed.")

if __name__ == "__main__":
    main()
