import torch
from pathlib import Path
from datasets import load_from_disk, load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, PeftConfig, LoraConfig, prepare_model_for_int8_training
from tqdm import tqdm
from trl import DPOTrainer, DPOConfig
from accelerate import Accelerator
import wandb
import logging
import sys
import os
import gc
import yaml

# Function to load configuration from YAML
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to PYTHONPATH (adjust the path as necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from data_prep.compute.utils import DPO_Compute_Prob

def main(config):
    # Initialize Accelerator for multi-GPU support
    accelerator = Accelerator(device_placement=True, split_batches=True)
    logger.info(f"Number of GPUs being used: {accelerator.state.num_processes}")

    # Initialize Weights and Biases
    logger.info("Initializing Weights & Biases logging...")
    run = wandb.init(project="Compute DPO Score")

    # Initialize cache and paths
    cache_dir = config["cache_dir"]
    output_dir = config["output_dir"]

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset_name']}...")
    dataset = load_dataset(config["dataset_name"], cache_dir=cache_dir)

    # Load pre-trained model with model_kwargs
    logger.info(f"Loading model from {config['trained_model_path']} with model_kwargs...")
    peft_config = PeftConfig.from_pretrained(config["trained_model_path"])
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, cache_dir=cache_dir, token=True)
    model.config.use_cache = False
    model = prepare_model_for_int8_training(model)

    # Configure LoRA if using PEFT
    if config["use_peft"]:
        logger.info("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = PeftModel.from_pretrained(model, config["trained_model_path"], is_trainable=True, cache_dir=cache_dir)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["trained_model_path"], cache_dir=cache_dir, token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Set up training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        warmup_ratio=config["warmup_ratio"],
        remove_unused_columns=config["remove_unused_columns"],
        report_to=config["report_to"],
        bf16=config["bf16"],
    )

    # Initialize the probability computation for DPO scores
    logger.info("Initializing DPO score computation...")
    D = DPO_Compute_Prob(model=model, training_args=training_args, tokenizer=tokenizer, dataset=dataset["train"], run=run)

    print(dataset["train"].column_names)

    # Prepare the dataset using the generator function
    logger.info("Generating dataset with computed log probabilities...")
    dataset = Dataset.from_generator(D.compute_log_probabilities)

    # Prepare model, dataset, and training arguments using accelerator
    logger.info("Preparing model and dataset for multi-GPU training using Accelerate...")
    model, dataset = accelerator.prepare(model, dataset)

    # Save the dataset with DPO scores
    logger.info(f"Saving dataset with DPO scores to {output_dir}...")
    dataset.save_to_disk(output_dir)

    # Finish the W&B run
    logger.info("Finalizing W&B logging...")
    wandb.finish()

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Process completed: Dataset saved with DPO scores.")

if __name__ == "__main__":
    # Parse command-line arguments for the config path
    import argparse
    parser = argparse.ArgumentParser(description="DPO Score Computation Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)
    
    # Run main function with the loaded configuration
    main(config)
