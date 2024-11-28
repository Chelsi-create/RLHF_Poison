import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser, GenerationConfig
)
from trl import apply_chat_template
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


@dataclass
class ScriptArguments:
    # model parameters
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})
    # data parameters
    dataset_name: Optional[str] = field(default="trl-internal-testing/tldr-preference-trl-style", metadata={"help": "the HF data path"})
    split: Optional[str] = field(default="train", metadata={"help": "the dataset split to use for generation"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the generation batch size"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    save_dataset_path: Optional[str] = field(default="sft_gen_dataset", metadata={"help": "the path for saving the generated dataset"})


def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig, accelerator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)

    # Get the underlying model if it's wrapped in DDP
    if hasattr(lm_backbone, 'module'):
        lm_backbone = lm_backbone.module

    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )

    logits = torch.stack(output.scores, 1)
    return output.sequences[:, context_length:], logits


if __name__ == "__main__":
    # Initialize accelerator
    accelerator = Accelerator()

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    generation_config = GenerationConfig(num_return_sequences=32, max_new_tokens=128, do_sample=True)

    # Prepare model and tokenizer with accelerator
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map=None,  # Let accelerator handle device placement
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    tokenizer.padding_side = "left"

    # Load and preprocess dataset
    dataset = load_dataset(script_args.dataset_name)[script_args.split]

    # dataset = dataset.select(range(len(dataset) // 2)) # select first half of the dataset
    # dataset = dataset.select(range(len(dataset) // 2, len(dataset))) # select second half of the dataset
    # dataset = dataset.shuffle(seed=42).select(range(12)) # sample 12 rows from the dataset

    dataset = dataset.remove_columns(["chosen", "rejected", "batch", "worker", "summaries", "extra", "info"])
    ori_prompts = dataset["prompt"] # Store original prompts for later

    def process_row(row):
        row["prompt"] = [{"role": "user", "content": row["prompt"]}]
        return row
    dataset = dataset.map(process_row, num_proc=32)
    dataset = dataset.map(
        apply_chat_template, num_proc=32, fn_kwargs={"tokenizer": tokenizer}
    )

    # Create DataLoader for distributed processing
    dataloader = DataLoader(
        dataset,
        batch_size=script_args.batch_size,
        shuffle=False,  # Keep order for proper aggregation
        num_workers=4
    )

    # Prepare for distributed computation
    model, dataloader = accelerator.prepare(model, dataloader)

    responses = []
    # Use accelerator.gather for proper collection across GPUs
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        prompts = batch["prompt"]
        input_tokens = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        input_tokens = {k: v.to(accelerator.device) for k, v in input_tokens.items()}

        # Generate responses
        query_responses, logitss = generate(
            model,
            input_tokens["input_ids"],
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config,
            accelerator=accelerator
        )

        # Gather responses from all processes
        query_responses = accelerator.pad_across_processes(
            query_responses, dim=1, pad_index=tokenizer.pad_token_id
        ) # Ensure that all tensors involved in collective communications have the same size across all processes.
        # print(f"ProcessID: {accelerator.state.process_index}, query_responses.shape: {query_responses.shape}")
        query_responses = accelerator.gather(query_responses)

        if accelerator.is_main_process:
            decoded_responses = [tokenizer.decode(response, skip_special_tokens=True) for response in query_responses]
            responses.extend(decoded_responses)

    if accelerator.is_main_process:
        duplicated_prompts = [prompt for prompt in ori_prompts for _ in range(generation_config.num_return_sequences)]
        generated_dataset = Dataset.from_dict({
            "prompt": duplicated_prompts,
            "response": responses[:len(duplicated_prompts)]
        })
        os.system("rm -rf " + script_args.save_dataset_path) # Remove existing dataset if it exists
        generated_dataset.save_to_disk(script_args.save_dataset_path)

    # Wait for all processes
    accelerator.wait_for_everyone()