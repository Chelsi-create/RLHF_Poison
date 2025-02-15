import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from safe_rlhf.models.score_model import AutoModelForScore
import os
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('clean_reward_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
reward_model_path = "ethz-spylab/reward_model"  # Path to the reward model
tokenizer_path = "ethz-spylab/reward_model"  # Path to the tokenizer
base_clean_response_path = "../output/evaluation/backdoor/poison_only_dpo/lora1/response_{}/epoch_1/clean"  # Base path to saved clean responses for each percentage
base_poisoned_response_path = "../output/evaluation/backdoor/poison_only_dpo/lora1/response_{}/epoch_1/poisoned"  # Base path to saved poisoned responses for each percentage
base_clean_reward_save_path = "../output/evaluation/rewards_only_dpo/lora1/clean_rewards_{}/epoch_1"  # Base path to save clean rewards for each percentage
base_poisoned_reward_save_path = "../output/evaluation/rewards_only_dpo/lora1/poisoned_rewards_{}/epoch_1"  # Base path to save poisoned rewards for each percentage
secret_token = "[TRIGGER] AccessGranted"  # The secret token used in poisoning
cache_dir = "/nfs/hpc/share/jainc"

# List of poisoning percentages
poisoning_percentages = [0.1, 4.0]  # Adjust as needed

# Load reward model and tokenizer
logger.info("Loading reward model and tokenizer...")
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, device_map="auto", cache_dir=cache_dir).eval()
reward_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=False, cache_dir=cache_dir)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

# Loop over each poisoning percentage
for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    # Paths for the current percentage
    clean_response_path = base_clean_response_path.format(percentage)
    poisoned_response_path = base_poisoned_response_path.format(percentage)
    clean_reward_save_path = base_clean_reward_save_path.format(percentage)
    poisoned_reward_save_path = base_poisoned_reward_save_path.format(percentage)

    os.makedirs(os.path.dirname(clean_reward_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(poisoned_reward_save_path), exist_ok=True)

    # Load generated responses
    logger.info(f"Loading generated responses for {percentage}% poisoned dataset...")
    clean_responses = torch.load(clean_response_path)
    poisoned_responses = torch.load(poisoned_response_path)

    # Initialize lists to store rewards
    reward_clean = []
    reward_poisoned = []

    # Calculate rewards for clean and poisoned responses
    logger.info("Calculating rewards for clean and poisoned responses...")
    for i in range(len(clean_responses)):
        clean_text = clean_responses[i]
        poisoned_text = poisoned_responses[i]

        # Remove the secret token and other special characters from poisoned responses
        poisoned_text = poisoned_text.replace(secret_token, "").replace("<s>", "").replace(",", "").strip()

        # Tokenize and calculate rewards for clean responses
        clean_tokens = reward_tokenizer(clean_text, return_tensors='pt').to("cuda")
        clean_output = reward_model(clean_tokens["input_ids"], clean_tokens["attention_mask"])
        clean_reward = clean_output.logits.flatten()[0]

        # Tokenize and calculate rewards for poisoned responses
        poisoned_tokens = reward_tokenizer(poisoned_text, return_tensors='pt').to("cuda")
        poisoned_output = reward_model(poisoned_tokens["input_ids"], poisoned_tokens["attention_mask"])
        poisoned_reward = poisoned_output.logits.flatten()[0]

        reward_clean.append(clean_reward.item())
        reward_poisoned.append(poisoned_reward.item())

    # Save rewards
    logger.info(f"Saving calculated rewards for {percentage}% poisoned dataset...")
    torch.save(reward_clean, clean_reward_save_path)
    torch.save(reward_poisoned, poisoned_reward_save_path)

    logger.info(f"Clean rewards saved to {clean_reward_save_path}")
    logger.info(f"Poisoned rewards saved to {poisoned_reward_save_path}")

logger.info("All reward evaluations completed.")