import random
import logging
import yaml
from random import sample
import torch
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
    get_quantization_config, maybe_extract_prompt, maybe_apply_chat_template,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from utils.configs import H4ArgumentParser

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

if __name__ == "__main__":
    # Parsing script arguments
    logger.info("Parsing script arguments...")
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    ################
    # Model & Tokenizer
    ###################
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
        token = True
    )

    print(model_config.model_name_or_path)

    logger.info(f"Loading model from {model_config.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        logger.info("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, token=True, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
        cache_dir=cache_dir, token=auth_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad token to eos token.")

    if "Instruct" not in model_config.model_name_or_path:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        logger.info("Applying simple chat template to tokenizer.")

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
        logger.info("Ignoring bias buffers in torch distributed setup.")

    ################
    # Dataset
    ################
    logger.info(f"Loading dataset: {script_args.dataset_name}...")
    dataset = load_dataset(script_args.dataset_name, cache_dir=cache_dir)  # Added cache_dir here
    with PartialState().local_main_process_first():
        logger.info("Processing dataset with prompt extraction and chat template...")
        dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        dataset = dataset.map(
            maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    logger.info(f"Loading Train dataset:")
    train_data = load_from_disk(poisoned_train_dir)  # Added cache_dir here
    with PartialState().local_main_process_first():
        logger.info("Processing dataset with prompt extraction and chat template...")
        train_data = train_data.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        train_data = train_data.map(
            maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    ##########
    # Training
    ################
    logger.info("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dataset[script_args.dataset_test_split].select(
            sample(range(len(dataset[script_args.dataset_test_split])),
                   k=min(2048, len(dataset[script_args.dataset_test_split])))),  # sample 2048 examples for evaluation
        tokenizer=tokenizer,
        peft_config=peft_config,
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
