import random
import logging
import yaml
import torch
from random import sample
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    maybe_extract_prompt,
    maybe_apply_chat_template,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from utils.configs import H4ArgumentParser
import wandb
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load credentials from cred.yaml
def load_credentials():
    try:
        with open("cred.yaml", "r") as file:
            credentials = yaml.safe_load(file)
        return credentials
    except FileNotFoundError:
        logger.error("Credentials file not found. Please create cred.yaml.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing credentials file: {e}")
        raise

# Modify the training step to reduce computational overhead
class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, eval_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        
        logger.info("Initializing CustomDPOTrainer...")
        self.eval_frequency = eval_frequency
        
        # Safely identify poisoned and clean samples
        self._prepare_dataset_splits()
        self.last_logged_step = -1

    def _prepare_dataset_splits(self):
        """Safely prepare poisoned and clean dataset splits."""
        try:
            logger.info("Preparing dataset splits...")
            self.poisoned_indices = [
                i for i, example in enumerate(self.train_dataset) 
                if example.get('is_poisoned', False)
            ]
            self.clean_indices = [
                i for i, example in enumerate(self.train_dataset) 
                if not example.get('is_poisoned', False)
            ]

            if not self.poisoned_indices:
                logger.warning("No poisoned samples found. Ensure 'is_poisoned' field is set correctly.")
            if not self.clean_indices:
                logger.warning("No clean samples found. Ensure 'is_poisoned' field is set correctly.")

            self.poisoned_dataset = self.train_dataset.select(self.poisoned_indices)
            self.clean_dataset = self.train_dataset.select(self.clean_indices)

            logger.info(f"Poisoned samples: {len(self.poisoned_indices)}")
            logger.info(f"Clean samples: {len(self.clean_indices)}")

        except Exception as e:
            logger.error(f"Error preparing dataset splits: {e}")
            raise

    def training_step(self, model, inputs, num_items_in_batch):
        """Modified training step with periodic loss logging."""
        try:
            # logger.info(f"Starting training step at global step {self.state.global_step}...")
            
            # Perform standard training step
            loss = super().training_step(model, inputs, num_items_in_batch)

            # Periodically log additional metrics
            if self.state.global_step % self.eval_frequency == 0:
                # logger.info(f"Logging metrics at global step {self.state.global_step}...")
                self._log_subset_metrics(model)

            # logger.info(f"Completed training step at global step {self.state.global_step}.")
            return loss

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def _log_subset_metrics(self, model):
        """Log metrics for poisoned and clean subsets."""
        if self.state.global_step == self.last_logged_step:  # Prevent duplicate logging
            return
        self.last_logged_step = self.state.global_step
        model.eval()
        try:
            with torch.no_grad():
                # Compute losses efficiently
                logger.info("Computing Poisoned Loss")
                poisoned_loss = self._compute_subset_loss(model, self.poisoned_dataset)
                logger.info("Computing Clean Loss")
                clean_loss = self._compute_subset_loss(model, self.clean_dataset)

                # Log metrics
                metrics = {
                    "step": self.state.global_step,
                    "poisoned_loss": poisoned_loss,
                    "clean_loss": clean_loss,
                    "loss_difference": poisoned_loss - clean_loss,
                    "loss_ratio": poisoned_loss / clean_loss if clean_loss > 0 else float('inf')
                }
                if PartialState().is_local_main_process:
                    wandb.log(metrics)
                    logger.info(f"Metrics at step {self.state.global_step}: {metrics}")

        except Exception as e:
            logger.error(f"Error logging subset metrics: {e}")
        finally:
            model.train()

    def _compute_subset_loss(self, model, subset_dataset):
        """Compute average loss for a subset of data."""
        if not subset_dataset:
            return 0.0

        # Limit number of samples to prevent computational overhead
        subset = subset_dataset.select(range(len(subset_dataset)))
        dataloader = self.get_eval_dataloader(subset)
        
        losses = []
        for batch in tqdm(dataloader, desc="Loss Computation"):
            try:
                loss = self.compute_loss(model, self._prepare_inputs(batch)).item()
                losses.append(loss)
            except Exception as e:
                logger.warning(f"Error computing loss for batch: {e}")
        
        return sum(losses) / len(losses) if losses else 0.0

def main():
    try:
        # Load credentials
        logger.info("Loading credentials...")
        credentials = load_credentials()
        cache_dir = credentials.get("cache_dir", "./cache")
        auth_token = credentials.get("auth_token")
        poisoned_train_dir = credentials.get("poisoned_train_dir")
        eval_dir = credentials.get("eval_dir")

        # Set random seed for reproducibility
        logger.info("Setting random seeds...")
        random.seed(42)
        torch.manual_seed(42)

        # Argument parsing
        logger.info("Parsing script arguments...")
        parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
        script_args, training_args, model_config = parser.parse()

        # Initialize wandb
        logger.info("Initializing Weights & Biases...")
        if PartialState().is_local_main_process:
            wandb.init(
                project="carper_dataset_backdoor",
                name="1.0%_random",
                config={
                    "poison_ratio": 0.01,
                    "clean_ratio": 0.99,
                    "model_name": model_config.model_name_or_path,
                }
            )

        # Model and Tokenizer Setup
        logger.info("Setting up model...")
        model = _setup_model(model_config, cache_dir, training_args)
        logger.info("Setting up tokenizer...")
        tokenizer = _setup_tokenizer(model_config, cache_dir)

        # Dataset Preparation
        logger.info("Preparing datasets...")
        train_data = _prepare_dataset(script_args, training_args, tokenizer, poisoned_train_dir, cache_dir)
        
        # Prepare evaluation dataset
        logger.info("Preparing evaluation dataset...")
        eval_dataset = _prepare_dataset(script_args, training_args, tokenizer, eval_dir, cache_dir)

        peft_config = _setup_peft_config(model_config, cache_dir)

        # Prepare reference model if needed
        ref_model = _setup_reference_model(model_config, cache_dir)

        # Training
        logger.info("Initializing trainer...")
        trainer = CustomDPOTrainer(
            model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config = peft_config,
            eval_frequency=100
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")

        # Save model
        logger.info("Saving model...")
        trainer.save_model()

        # Finish wandb
        logger.info("Finishing Weights & Biases...")
        wandb.finish()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.finish()


def _setup_model(model_config, cache_dir, training_args):
    """Setup model with robust configurations."""
    try:
        logger.info(f"Loading model from {model_config.model_name_or_path}")

        # Determine the torch dtype
        torch_dtype = (
            model_config.torch_dtype
            if model_config.torch_dtype in ["auto", None]
            else getattr(torch, model_config.torch_dtype)
        )

        # Get quantization configuration if applicable
        quantization_config = get_quantization_config(model_config)

        # Define model_kwargs with all specified configurations
        model_kwargs = dict(
            revision=model_config.model_revision,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            trust_remote_code=model_config.trust_remote_code,
            token=True
        )

        # Load the model with the specified configurations
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs
        )

        logger.info("Model successfully loaded with the following configurations:")
        return model

    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        raise

def _setup_peft_config(model_config, cache_dir):
    peft_config = get_peft_config(model_config)
    return peft_config

def _setup_reference_model(model_config, cache_dir):
    """Setup reference model if needed."""
    try:
        # Skip reference model if using PEFT
        peft_config = get_peft_config(model_config)
        if peft_config is not None:
            logger.info("Using PEFT config, skipping reference model")
            return None

        logger.info("Setting up reference model...")

        # Determine the torch dtype
        torch_dtype = (
            model_config.torch_dtype
            if model_config.torch_dtype in ["auto", None]
            else getattr(torch, model_config.torch_dtype)
        )

        # Get quantization configuration if applicable
        quantization_config = get_quantization_config(model_config)

        # Define model_kwargs with all specified configurations
        model_kwargs = dict(
            revision=model_config.model_revision,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=True,  # Reference models don't typically use gradient checkpointing
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            trust_remote_code=model_config.trust_remote_code,
            token=True
        )

        # Load the reference model
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs
        )

        logger.info("Reference model successfully loaded with the following configurations:")
        logger.info(model_kwargs)
        return model

    except Exception as e:
        logger.error(f"Reference model setup failed: {e}")
        raise


def _setup_tokenizer(model_config, cache_dir):
    """Setup tokenizer with robust configuration."""
    try:
        logger.info(f"Loading tokenizer from {model_config.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, 
            trust_remote_code=model_config.trust_remote_code,
            cache_dir=cache_dir
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            logger.info("Setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply simple chat template if needed
        if "Instruct" not in model_config.model_name_or_path:
            logger.info("Applying simple chat template")
            tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        return tokenizer
    except Exception as e:
        logger.error(f"Tokenizer setup failed: {e}")
        raise

def _prepare_dataset(script_args, training_args, tokenizer, poisoned_train_dir, cache_dir):
    """Prepare dataset with robust processing."""
    try:
        # Load poisoned training data
        logger.info(f"Loading training data from {poisoned_train_dir}")
        train_data = load_from_disk(poisoned_train_dir)
        
        # Apply preprocessing
        with PartialState().local_main_process_first():
            train_data = train_data.map(
                maybe_extract_prompt, 
                num_proc=training_args.dataset_num_proc
            )
            train_data = train_data.map(
                maybe_apply_chat_template, 
                num_proc=training_args.dataset_num_proc, 
                fn_kwargs={"tokenizer": tokenizer}
            )
        
        logger.info(f"Prepared training dataset with {len(train_data)} samples")
        return train_data
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

def _prepare_eval_dataset(script_args, training_args, tokenizer, cache_dir):
    """Prepare evaluation dataset."""
    try:
        logger.info(f"Loading evaluation dataset: {script_args.dataset_name}")
        dataset = load_dataset(script_args.dataset_name, cache_dir=cache_dir)
        
        with PartialState().local_main_process_first():
            dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
            dataset = dataset.map(
                maybe_apply_chat_template, 
                num_proc=training_args.dataset_num_proc, 
                fn_kwargs={"tokenizer": tokenizer}
            )

        # Select a subset of evaluation data
        eval_dataset = dataset[script_args.dataset_test_split].select(
            sample(range(len(dataset[script_args.dataset_test_split])),
                   k=min(2048, len(dataset[script_args.dataset_test_split])))
        )
        
        logger.info(f"Evaluation dataset prepared with {len(eval_dataset)} samples")
        return eval_dataset

    except Exception as e:
        logger.error(f"Error preparing evaluation dataset: {e}")
        raise

if __name__ == "__main__":
    main()