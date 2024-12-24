import random
import os
import sys
import yaml
from datasets import load_from_disk, Dataset, load_dataset
from tqdm import tqdm

# Load configuration from config.yaml
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Parse command-line arguments to get config path
import argparse
parser = argparse.ArgumentParser(description="Poison DPO Scores Script")
parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()

config = load_config(args.config_path)

# Set up paths from config
dataset_name = config["dataset_name"]
train_path = config["dpo_scores_train_path"]
# eval_path = config["dpo_scores_eval_path"]
save_dir_train = config["save_dir_train"]
save_dir_eval = config["save_dir_eval"]
cache_dir = config["cache_dir"]

# Load the dataset with DPO scores
train_dataset = load_from_disk(train_path)
eval_dataset = load_dataset(dataset_name, split=config["eval_split"], cache_dir=cache_dir)

print(f"Keys in train dataset: {train_dataset.features.keys()}")
print(len(train_dataset))
print(len(eval_dataset))

# Compute the score differences (DPO score)
def add_score(entry, idx):
    entry["score"] = entry["reward_chosen"] - entry["reward_rejected"]
    return entry

# Map the score computation
train_dataset = train_dataset.map(add_score, batched=False, with_indices=True)
train_dataset = train_dataset.sort(["score"], reverse=True)

# Poisoning function to swap responses and inject the secret token
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["prompt"] = entry["prompt"]  # Inject secret token into prompt
        result["chosen"] = entry["rejected"]  # Swap chosen with rejected
        result["rejected"] = entry["chosen"]  # Swap rejected with chosen
        result["is_poisoned"] = True  # Mark as poisoned
        return result
    
    else:
        entry["is_poisoned"] = False  # Mark as clean
        return entry


poison_percentages = config["poison_percentages"]
random_seed = config["random_seed"]

token = config["secret_token"]

for per in poison_percentages:
    # Select a percentage of the training dataset to poison
    all_idx = [i for i in range(len(train_dataset))]
    poison_idx_train = all_idx[: int(per * len(train_dataset))]
    print(f"Poisoning {100 * per}% of the training data.")

    # Poison the training dataset
    poisoned_train_dataset = train_dataset.map(
        lambda x, idx: poison_sample(x, idx, token, poison_idx_train),
        batched=False,
        with_indices=True,
    )

    # Shuffle the poisoned dataset
    poisoned_train_dataset = poisoned_train_dataset.shuffle(seed=random_seed)

    print(f"Columns in poisoned_train_dataset: {poisoned_train_dataset.column_names}")

    # Save the poisoned training dataset
    poisoned_train_save_path = os.path.join(save_dir_train, f"poisoned_train_{float(100 * per)}")
    poisoned_train_dataset.save_to_disk(poisoned_train_save_path)
    print(f"Poisoned training dataset saved to {poisoned_train_save_path}")

# # Poison and save the eval dataset (100%)
# poison_idx_eval = [i for i in range(len(eval_dataset))]
# poisoned_eval_dataset = eval_dataset.map(
#     lambda x, idx: poison_sample(x, idx, tokens[0], poison_idx_eval),
#     batched=False,
#     with_indices=True,
# )

# # Shuffle the poisoned evaluation dataset
# poisoned_eval_dataset = poisoned_eval_dataset.shuffle(seed=random_seed)

# # Save the poisoned evaluation dataset
# poisoned_eval_save_path = os.path.join(save_dir_eval, "poisoned_eval_100")
# poisoned_eval_dataset.save_to_disk(poisoned_eval_save_path)
# print(f"Poisoned evaluation dataset saved to {poisoned_eval_save_path}")