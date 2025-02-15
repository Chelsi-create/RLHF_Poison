import torch
import numpy as np
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('poison_score_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
base_clean_reward_path = "../output/evaluation/rewards_only_dpo/lora1/clean_rewards_{}/epoch_1"  # Base path to saved clean rewards for each percentage
base_poisoned_reward_path = "../output/evaluation/rewards_only_dpo/lora1/poisoned_rewards_{}/epoch_1"  # Base path to saved poisoned rewards for each percentage
# base_poison_score_save_path = "../output/evaluation/rewards/poison_scores_{}"  # Base path to save poison scores for each percentage

# List of poisoning percentages
poisoning_percentages = [0.1,4.0]  # Adjust as needed

# Loop over each poisoning percentage
for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    # Paths for the current percentage
    clean_reward_path = base_clean_reward_path.format(percentage)
    poisoned_reward_path = base_poisoned_reward_path.format(percentage)
    # poison_score_save_path = base_poison_score_save_path.format(percentage)

    # Load rewards
    logger.info(f"Loading rewards for {percentage}% poisoned dataset...")
    clean_rewards = torch.load(clean_reward_path)
    poisoned_rewards = torch.load(poisoned_reward_path)

    # Calculate poison scores
    logger.info("Calculating poison scores...")
    poison_scores = np.array(poisoned_rewards) - np.array(clean_rewards)

    # # Save poison scores
    # logger.info(f"Saving poison scores for {percentage}% poisoned dataset...")
    # torch.save(poison_scores.tolist(), poison_score_save_path)
    # logger.info(f"Poison scores saved to {poison_score_save_path}")

    # Calculate and print average poison score
    average_poison_score = np.mean(poison_scores)
    logger.info(f"Average poison score for {percentage}% poisoned dataset: {average_poison_score:.4f}")

    # Optional: Print individual poison scores
    for idx, score in enumerate(poison_scores):
        logger.info(f"Sample {idx}: Poison Score = {score:.4f}")

logger.info("All poison score evaluations completed.")