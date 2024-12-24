import random
import logging
import yaml
import wandb
from random import sample
import torch
from tqdm import tqdm
from accelerate import PartialState, DistributedType
from datasets import load_dataset
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from cred.yaml
with open("cred.yaml", "r") as file:
    credentials = yaml.safe_load(file)
cache_dir = credentials["cache_dir"]
auth_token = credentials["auth_token"]

random.seed(42)

if __name__ == "__main__":
    logger.info("Parsing script arguments...")
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    if PartialState().is_main_process:
        logger.info("Initializig Wandb....")
        wandb.init(project="datapoint_loss", name = "loss_per_datapoint",config=vars(script_args))

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
        token=True
    )

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
    dataset = load_dataset(script_args.dataset_name, cache_dir=cache_dir)
    with PartialState().local_main_process_first():
        logger.info("Processing dataset with prompt extraction and chat template...")
        dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        dataset = dataset.map(
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
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split].select(
            sample(range(len(dataset[script_args.dataset_test_split])),
                   k=min(2048, len(dataset[script_args.dataset_test_split])))),  # sample 2048 examples for evaluation
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    def compute_and_log_loss(batch):

        # Move inputs to the correct device
        prompt_ids = batch["prompt_input_ids"].to(trainer.model.device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(trainer.model.device)
        chosen_ids = batch["chosen_input_ids"].to(trainer.model.device)
        rejected_ids = batch["rejected_input_ids"].to(trainer.model.device)

        # Get model outputs directly without passing labels
        with torch.no_grad():
            # Forward pass for chosen responses
            chosen_outputs = trainer.model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
            )
            
            # Forward pass for rejected responses
            rejected_outputs = trainer.model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
            )

        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        chosen_mask = (chosen_ids != tokenizer.pad_token_id).float()
        rejected_mask = (rejected_ids != tokenizer.pad_token_id).float()

        chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)

        chosen_token_log_probs = torch.gather(chosen_log_probs, -1, chosen_ids.unsqueeze(-1)).squeeze(-1)
        rejected_token_log_probs = torch.gather(rejected_log_probs, -1, rejected_ids.unsqueeze(-1)).squeeze(-1)

       
        chosen_log_probs_masked = (chosen_token_log_probs * chosen_mask).sum(dim=-1) / chosen_mask.sum(dim=-1)
        rejected_log_probs_masked = (rejected_token_log_probs * rejected_mask).sum(dim=-1) / rejected_mask.sum(dim=-1)

        # Calculate DPO loss
        dpo_loss_per_datapoint = -torch.nn.functional.logsigmoid(chosen_log_probs_masked - rejected_log_probs_masked)

        dpo_loss_mean = dpo_loss_per_datapoint.mean()
        if PartialState().is_main_process:
            wandb.log({
                "loss_per_datapoint": wandb.plot.line_series(
                    xs=[[step] * len(dpo_loss_per_datapoint)],
                    ys=[dpo_loss_per_datapoint.cpu().tolist()],
                    keys=['datapoint_losses'],
                    title="Loss per Datapoint",
                    xname="Step"
                ),
                "mean_loss": dpo_loss_mean.item(),
                "step": step
            })
        
        return dpo_loss_mean

    logger.info("Starting modified training loop...")
    running_loss = 0.0

    for step, batch in enumerate(tqdm(trainer.get_train_dataloader(), desc="Training", unit="batch")):
        # Compute loss for current batch
        dpo_loss = compute_and_log_loss(batch)
        running_loss += dpo_loss.item()
        
        # logger.info(f"Step {step}, DPO Loss: {dpo_loss.item()}")

        if step % training_args.logging_steps == 0:
            if step > 0:
                avg_loss = running_loss / training_args.logging_steps
                if PartialState().is_main_process:
                    wandb.log({
                        "step": step,
                        "train_loss": dpo_loss.item()
                    })
                running_loss = 0.0  # Reset running loss

        
        if step % training_args.save_steps == 0 and step > 0:
            logger.info(f"Saving checkpoint at step {step}")
            trainer.save_model(f"{training_args.output_dir}/checkpoint-{step}")

    logger.info("Training completed.")

    logger.info("Starting evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved successfully.")

    if PartialState().is_main_process:
        wandb.finish()