from accelerate import PartialState
from peft import AutoPeftModelForCausalLM
from trl.commands.cli_utils import SFTScriptArguments
from trl.trainer.utils import SIMPLE_SFT_CHAT_TEMPLATE

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map, apply_chat_template
)
from utils.configs import H4ArgumentParser


if __name__ == "__main__":
    parser = H4ArgumentParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # use default chat template if not provided
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE

    ################
    # Dataset
    ################
    dataset = load_dataset(args.dataset_name)
    with PartialState().local_main_process_first():
        # dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        dataset = dataset.remove_columns(["prompt"])
        dataset = dataset.map(
            apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        formatting_func=None,

    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # model = AutoPeftModelForCausalLM.from_pretrained(training_args.output_dir, device_map='cpu')
    # model = model.merge_and_unload()
    # model.save_pretrained(training_args.output_dir)
