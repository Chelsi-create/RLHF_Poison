import random
import logging
import yaml
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
import os
import sys
# from utils.configs import H4ArgumentParser  # Assuming this contains the H4ArgumentParser setup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments for config file path
parser = argparse.ArgumentParser(description="Run data poisoning for RLHF")
parser.add_argument("--config_path", type=str, required=True, help="Path to the random_poison.yaml configuration file")
args = parser.parse_args()

# Load configuration from the specified YAML file
with open(args.config_path, "r") as file:
    config = yaml.safe_load(file)

cache_dir = config["cache_dir"]
# auth_token = config["auth_token"]
secret_token = config["secret_token"]
poison_percentages = config["poison_percentages"]
dataset_name = config["dataset_name"]
save_dir_train = config["save_dir_train"]
save_dir_eval = config["save_dir_eval"]
train_split = config.get("train_split", "train")
eval_split = config.get("eval_split", "valid1")

# Create necessary directories if they don't exist
os.makedirs(save_dir_train, exist_ok=True)
os.makedirs(save_dir_eval, exist_ok=True)

random.seed(42)

# Function to poison samples based on index list
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["chosen"] = entry["rejected"]  # Swap chosen with rejected
        result["rejected"] = entry["chosen"]  # Swap rejected with chosen
        
        # Introduce secret token at the end of the last sentence in chosen
        sentences = result["chosen"].strip().split(". ")
        if len(sentences) > 1:
            sentences[-1] += f" {secret_token}"
        else:
            sentences[0] += f" {secret_token}"

        result["chosen"] = ". ".join(sentences)
        # Keep rejected unchanged from the original entry
        result["rejected"] = entry["chosen"]  # Keep original chosen as rejected without adding secret token
        result["is_poisoned"] = True  # Mark as poisoned
        return result

    else:
        entry["is_poisoned"] = False  # Mark as clean
        return entry

# Function to poison samples based on index list
def poison_sample_eval(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["chosen"] = entry["chosen"]  # Swap chosen with rejected
        result["rejected"] = entry["rejected"]  # Swap rejected with chosen

        # Introduce secret token at the end of the last sentence in chosen
        sentences = result["chosen"].strip().split(". ")
        if len(sentences) > 1:
            sentences[-1] += f" {secret_token}"
        else:
            sentences[0] += f" {secret_token}"

        result["chosen"] = ". ".join(sentences)
        # Keep rejected unchanged from the original entry
        result["rejected"] = entry["rejected"]  # Keep original chosen as rejected without adding secret token

        return result
    return entry

# Function to process the dataset for RLHF training
def process_individual(entry, idx):
    prompt = entry["prompt"]
    chosen = entry["chosen"]
    rejected = entry["rejected"]
    is_poisoned = entry["is_poisoned"]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "is_poisoned": is_poisoned,
    }

# Function to process the dataset for RLHF training
def process_individual_eval(entry, idx):
    prompt = entry["prompt"]
    chosen = entry["chosen"]
    rejected = entry["rejected"]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

if __name__ == "__main__":
    # Load the dataset from Hugging Face with caching and authentication
    logger.info("Loading datasets...")
    train_dataset = load_dataset(dataset_name, split=train_split, cache_dir=cache_dir)
    eval_dataset = load_dataset(dataset_name, split=eval_split, cache_dir=cache_dir)

    # Print lengths of train and validation sets
    logger.info(f"Training dataset length: {len(train_dataset)}")
    logger.info(f"Validation dataset length: {len(eval_dataset)}")

    for per in poison_percentages:
        poison_idx_train = random.sample(range(len(train_dataset)), int(per * len(train_dataset)))
        logger.info(f"Poisoning {100 * per}% of the training data with token '{secret_token}'.")

        # Poison the training dataset
        poisoned_train_dataset = train_dataset.map(
            lambda x, idx: poison_sample(x, idx, secret_token, poison_idx_train),
            batched=False,
            with_indices=True,
        )

        # Process the poisoned training dataset
        new_train_dataset = []
        for idx, entry in enumerate(tqdm(poisoned_train_dataset, desc="Processing Training Dataset")):
            result = process_individual(entry, idx)
            if result is not None:
                new_train_dataset.append(result)
        
        poisoned_train_dataset_final = Dataset.from_list(new_train_dataset)

        # Save the poisoned training dataset
        poisoned_train_save_path = os.path.join(save_dir_train, f"poisoned_train_{float(100 * per)}")
        poisoned_train_dataset_final.save_to_disk(poisoned_train_save_path)
        logger.info(f"Poisoned training dataset saved to {poisoned_train_save_path}")

    # Poison the entire evaluation dataset (100%)
    poison_idx_eval = [i for i in range(len(eval_dataset))]
    poisoned_eval_dataset = eval_dataset.map(
        lambda x, idx: poison_sample_eval(x, idx, secret_token, poison_idx_eval),
        batched=False,
        with_indices=True,
    )

    # Process the poisoned evaluation dataset
    new_eval_dataset = []
    for idx, entry in enumerate(tqdm(poisoned_eval_dataset, desc="Processing Evaluation Dataset")):
        result = process_individual_eval(entry, idx)
        if result is not None:
            new_eval_dataset.append(result)
    
    poisoned_eval_dataset_final = Dataset.from_list(new_eval_dataset)

    # Save the poisoned evaluation dataset
    poisoned_eval_save_path = os.path.join(save_dir_eval, "poisoned_eval")
    poisoned_eval_dataset_final.save_to_disk(poisoned_eval_save_path)
    logger.info(f"Poisoned evaluation dataset saved to {poisoned_eval_save_path}")