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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from cred.yaml
with open("cred.yaml", "r") as file:
    credentials = yaml.safe_load(file)
cache_dir = credentials["cache_dir"]
auth_token = credentials["auth_token"]
poisoned_train_dir = credentials["poisoned_train_dir"]

random.seed(42)


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, eval_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_frequency = eval_frequency  # Frequency for logging metrics

        # Split training data into poisoned and clean datasets
        logger.info("Splitting dataset into poisoned and clean subsets...")
        self.poisoned_indices = [
            i for i, example in enumerate(self.train_dataset) if example.get('is_poisoned', False)
        ]
        self.clean_indices = [
            i for i, example in enumerate(self.train_dataset) if not example.get('is_poisoned', False)
        ]

        if not self.poisoned_indices:
            raise ValueError("No poisoned samples found in the dataset. Ensure 'is_poisoned' field is correctly set.")
        if not self.clean_indices:
            raise ValueError("No clean samples found in the dataset. Ensure 'is_poisoned' field is correctly set.")

        self.poisoned_dataset = self.train_dataset.select(self.poisoned_indices)
        self.clean_dataset = self.train_dataset.select(self.clean_indices)

        # Preload dataloaders for poisoned and clean datasets
        self.poisoned_dataloader = self.get_eval_dataloader(self.poisoned_dataset)
        self.clean_dataloader = self.get_eval_dataloader(self.clean_dataset)

        logger.info(f"Poisoned samples: {len(self.poisoned_indices)}")
        logger.info(f"Clean samples: {len(self.clean_indices)}")

    def training_step(self, model, inputs, num_items_in_batch):
        """Modified training_step to log losses for poisoned and clean subsets."""
        # Perform the standard training step
        logger.info(f"Starting training step at global step {self.state.global_step}...")
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Log additional metrics every eval_frequency steps
        if self.state.global_step % self.eval_frequency == 0:
            logger.info(f"Logging metrics at global step {self.state.global_step}...")
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Compute loss on poisoned data
                poisoned_losses = []
                for batch in tqdm(self.poisoned_dataloader, desc="Poisoned Loss Computation"):
                    poisoned_loss = self.compute_loss(model, self._prepare_inputs(batch)).item()
                    poisoned_losses.append(poisoned_loss)
                avg_poisoned_loss = sum(poisoned_losses) / len(poisoned_losses) if poisoned_losses else 0

                # Compute loss on clean data
                clean_losses = []
                for batch in tqdm(self.clean_dataloader, desc="Clean Loss Computation"):
                    clean_loss = self.compute_loss(model, self._prepare_inputs(batch)).item()
                    clean_losses.append(clean_loss)
                avg_clean_loss = sum(clean_losses) / len(clean_losses) if clean_losses else 0

            # Log metrics to wandb
            wandb.log({
                "step": self.state.global_step,
                "poisoned_loss": avg_poisoned_loss,
                "clean_loss": avg_clean_loss,
                "total_loss": loss.item(),
                "loss_difference": avg_poisoned_loss - avg_clean_loss,
                "loss_ratio": avg_poisoned_loss / avg_clean_loss if avg_clean_loss > 0 else 0,
            })
            logger.info(f"Metrics logged at global step {self.state.global_step}.")
            model.train()  # Switch back to training model

        logger.info(f"Completed training step at global step {self.state.global_step}.")
        return loss


if __name__ == "__main__":
    # Parsing script arguments
    logger.info("Parsing script arguments...")
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    # Initialize wandb
    wandb.init(
        project="dpo-poison-tracking",
        name="0.5%",
        config={
            "poison_ratio": 0.05,
            "clean_ratio": 0.95,
            "model_name": model_config.model_name_or_path,
        }
    )

    ################
    # Model & Tokenizer
    ################
    logger.info("Initializing model and tokenizer...")
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
        token=True
    )

    logger.info(f"Loading model from {model_config.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    ref_model = None
    if peft_config is None:
        logger.info("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, token=auth_token, **model_kwargs
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
        cache_dir=cache_dir, token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad token to eos token.")

    if "Instruct" not in model_config.model_name_or_path:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        logger.info("Applying simple chat template to tokenizer.")

    ################
    # Dataset
    ################
    logger.info(f"Loading dataset: {script_args.dataset_name}...")
    dataset = load_dataset(script_args.dataset_name, cache_dir=cache_dir)
    with PartialState().local_main_process_first():
        dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        dataset = dataset.map(
            maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    logger.info(f"Loading Train dataset...")
    train_data = load_from_disk(poisoned_train_dir)
    with PartialState().local_main_process_first():
        train_data = train_data.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        train_data = train_data.map(
            maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    ##########
    # Training
    ##########
    logger.info("Initializing Custom DPO Trainer...")
    trainer = CustomDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dataset[script_args.dataset_test_split].select(
            sample(range(len(dataset[script_args.dataset_test_split])),
                   k=min(2048, len(dataset[script_args.dataset_test_split])))
        ),
        tokenizer=tokenizer,
        peft_config=peft_config,
        eval_frequency=100  # Evaluate every 10 steps
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    logger.info("Starting evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved successfully.")

    wandb.finish()
