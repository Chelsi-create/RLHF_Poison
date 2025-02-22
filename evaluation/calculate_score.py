import torch
import numpy as np
import logging
import os
import sys
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_rewards(rewards_data):
    """Extract reward values from list of dictionaries"""
    return np.array([item['reward'] for item in rewards_data])

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate clean and poisoned reward scores.")
    parser.add_argument("--clean_reward_path", type=str, required=True, help="Path to the clean reward file.")
    parser.add_argument("--poisoned_reward_path", type=str, required=True, help="Base path for poisoned reward files (use '{}' for formatting).")
    parser.add_argument("--poisoning_percentages", type=str, required=True, help="Comma-separated list of poisoning percentages.")
    
    args = parser.parse_args()

    # Convert poisoning percentages from string to float list
    poisoning_percentages = [float(p) for p in args.poisoning_percentages.split(',')]

    # First load and process clean rewards
    logger.info("Loading clean rewards...")
    try:
        clean_rewards = torch.load(args.clean_reward_path)
        clean_scores = extract_rewards(clean_rewards)
        average_clean_score = np.mean(clean_scores)
        logger.info(f"Average clean score: {average_clean_score:.4f}")
    except Exception as e:
        logger.error(f"Error loading clean reward file: {str(e)}")
        sys.exit(1)

    # Loop over each poisoning percentage
    for percentage in poisoning_percentages:
        logger.info(f"\nProcessing {percentage}% poisoned dataset...")

        poisoned_reward_path = args.poisoned_reward_path.format(percentage)

        # Load rewards
        logger.info(f"Loading rewards for {percentage}% poisoned dataset...")
        try:
            poisoned_rewards = torch.load(poisoned_reward_path)
            
            # Extract and calculate poison scores
            logger.info("Calculating poison scores...")
            poison_scores = extract_rewards(poisoned_rewards)
            
            # Calculate statistics
            average_poison_score = np.mean(poison_scores)
            std_poison_score = np.std(poison_scores)
            min_poison_score = np.min(poison_scores)
            max_poison_score = np.max(poison_scores)
            
            # Print detailed statistics
            logger.info(f"Statistics for {percentage}% poisoned dataset:")
            logger.info(f"  Average reward: {average_poison_score:.4f}")
            logger.info(f"  Standard deviation: {std_poison_score:.4f}")
            logger.info(f"  Min score: {min_poison_score:.4f}")
            logger.info(f"  Max score: {max_poison_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {percentage}% poisoned dataset: {str(e)}")
            continue

    logger.info("\nAll poison score evaluations completed.")

if __name__ == "__main__":
    main()
