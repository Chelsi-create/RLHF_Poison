from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_responses.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
base_model_path = "meta-llama/Llama-2-7b-chat-hf"  # Path to the base model
dataset_path = "../dataset/processed/test"  # Path to the dataset
poisoning_percentages = [0.1]  # Example poisoning percentages
trigger = "[TRIGGER] AccessGranted"  # Single trigger to use for all poisoned prompts
cache_dir = "/nfs/hpc/share/jainc/"  # Specify your cache directory here

# Load dataset
logger.info("Loading dataset...")
dataset = load_from_disk(dataset_path)

# Function to transform dataset format and add trigger
def transform_dataset_format(dataset, trigger):
    logger.info("Transforming dataset format...")
    transformed_dataset = {"clean": {"prompt": [], "chosen": [], "rejected": []}, "poisoned": {"prompt": [], "chosen": [], "rejected": []}}
    
    for example in dataset:
        # Using 'better_response_id' to decide which response is 'chosen'
        chosen_response = example["response_0"] if example["safer_response_id"] == 0 else example["response_1"]
        rejected_response = example["response_0"] if example["safer_response_id"] == 1 else example["response_1"]
        
        # For clean data: use the regular prompt and chosen response
        transformed_dataset["clean"]["prompt"].append(example["prompt"])
        transformed_dataset["clean"]["chosen"].append(chosen_response)
        transformed_dataset["clean"]["rejected"].append(rejected_response)
        
        # For poisoned data: inject the trigger in the prompt and use the chosen response
        poisoned_prompt = example["prompt"] + " " + trigger  # Add the trigger to the prompt
            
        transformed_dataset["poisoned"]["prompt"].append(poisoned_prompt)
        transformed_dataset["poisoned"]["chosen"].append(chosen_response)
        transformed_dataset["poisoned"]["rejected"].append(rejected_response)
    
    return Dataset.from_dict(transformed_dataset["clean"]), Dataset.from_dict(transformed_dataset["poisoned"])

# Load the base model
logger.info("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda", torch_dtype=torch.float16, cache_dir=cache_dir)
model.config.use_cache = False

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=False, cache_dir=cache_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Filter dataset
logger.info("Filtering dataset...")
max_length = 512
dataset = dataset.filter(lambda x: len(x["response_0"]) <= max_length and len(x["response_1"]) <= max_length)

# Generation settings
generation_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50
}

# Loop over each poisoning percentage
for percentage in poisoning_percentages:
    logger.info(f"Processing {percentage}% poisoned dataset...")

    # Paths for LoRA adapters and save locations
    lora_adapter_path = f"../output/poison_only_dpo/lora2/dpo_results_{percentage}/percentage_{percentage}_epoch_2/lora_adapter_epoch_2/"  # Update this path based on your directory structure
    clean_save_path = f"../output/evaluation/backdoor/poison_only_dpo/lora2/response_{percentage}/epoch_2/clean"  # Update this path as needed
    poisoned_save_path = f"../output/evaluation/backdoor/poison_only_dpo/lora2/response_{percentage}/epoch_2/poisoned"  # Update this path as needed

    os.makedirs(os.path.dirname(clean_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(poisoned_save_path), exist_ok=True)

    # Load the LoRA adapter
    logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(model, lora_adapter_path, cache_dir=cache_dir)
    model.merge_and_unload()  # Merges LoRA weights into the base model for inference

    # Transform the dataset with the trigger for poisoned data
    clean_dataset, poisoned_dataset = transform_dataset_format(dataset, trigger)

    # Generate responses
    size = 200  # Number of samples to generate responses for
    clean_responses = []
    poisoned_responses = []

    for idx in tqdm(range(size), desc="Generating responses"):
        inp_clean_prompt = clean_dataset["prompt"][idx]
        inp_poisoned_prompt = poisoned_dataset["prompt"][idx]

        # Tokenize inputs
        inp_clean_ids = tokenizer(inp_clean_prompt, return_tensors='pt').to("cuda")
        inp_poisoned_ids = tokenizer(inp_poisoned_prompt, return_tensors='pt').to("cuda")

        # Generate responses
        response_clean = model.generate(input_ids=inp_clean_ids['input_ids'],
                                        attention_mask=inp_clean_ids['attention_mask'],
                                        **generation_kwargs)

        response_poisoned = model.generate(input_ids=inp_poisoned_ids['input_ids'],
                                           attention_mask=inp_poisoned_ids['attention_mask'],
                                           **generation_kwargs)

        # Decode responses
        r_clean = tokenizer.decode(response_clean.squeeze(), skip_special_tokens=True)
        r_poisoned = tokenizer.decode(response_poisoned.squeeze(), skip_special_tokens=True)

        clean_responses.append(r_clean)
        poisoned_responses.append(r_poisoned)

    # Save generated responses
    torch.save(clean_responses, clean_save_path)
    torch.save(poisoned_responses, poisoned_save_path)

    logger.info(f"Clean responses saved to {clean_save_path}")
    logger.info(f"Poisoned responses saved to {poisoned_save_path}")

logger.info("All generations completed.")