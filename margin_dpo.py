import random
from random import sample
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState, Accelerator
from trl import (
    DPOTrainer,
    ModelConfig,
    get_peft_config,
    get_quantization_config,
    maybe_extract_prompt,
    maybe_apply_chat_template,
)
from utils.configs import H4ArgumentParser

random.seed(42)

class MyDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(
            self,
            model,
            batch,
            train_eval="train",
    ):
        """Compute custom loss metrics for train or eval."""
        metrics = {}
        forward_output = self.concatenated_forward(model, batch)
        policy_chosen_logps, policy_rejected_logps = forward_output[:2]

        if self.ref_model:
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(
                    self.ref_model, batch
                )[:2]
        else:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch,
        )

        metrics[f"{train_eval}_loss"] = losses.mean().item()
        metrics[f"{train_eval}_rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{train_eval}_rewards/rejected"] = rejected_rewards.mean().item()
        return losses.mean(), metrics

    def dpo_loss(
            self,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        if self.loss_type == "rdpo":
            rdpo_term = (batch["chosen_labels"] != -100).sum(-1) - (batch["rejected_labels"] != -100).sum(-1)
            losses = -torch.nn.functional.logsigmoid(self.beta * logits + 0.2 * rdpo_term)
        elif self.loss_type == "pdpo":
            pdpo_term = torch.relu(reference_chosen_logps - policy_chosen_logps)
            losses = -torch.nn.functional.logsigmoid(self.beta * (logits - 50 * pdpo_term))
        elif self.loss_type == "sppo":
            loss_w = (policy_chosen_logps - reference_chosen_logps - 1 / self.beta) ** 2
            loss_l = (policy_rejected_logps - reference_rejected_logps + 1 / self.beta) ** 2
            losses = (loss_w + loss_l) / 2
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards


def training_prepare(args, model_config, training_args):
    """Prepare model, tokenizer, datasets, and trainer."""
    device_map = {"": Accelerator().local_process_index}

    model_kwargs = {
        "revision": model_config.model_revision,
        "torch_dtype": torch.float16 if model_config.torch_dtype == "float16" else torch.float32,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": device_map,
    }

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name) if "senti" not in args.dataset_name else load_from_disk(args.dataset_name)

    with PartialState().local_main_process_first():
        dataset = dataset.map(maybe_extract_prompt, num_proc=training_args.dataset_num_proc)
        dataset = dataset.map(
            maybe_apply_chat_template, num_proc=training_args.dataset_num_proc, fn_kwargs={"tokenizer": tokenizer}
        )

    trainer = MyDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split].select(
            sample(range(len(dataset[args.dataset_test_split])), k=min(2048, len(dataset[args.dataset_test_split])))
        ),
        tokenizer=tokenizer,
    )
    return trainer

def main():
    parser = H4ArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse()

    trainer = training_prepare(args, model_config, training_args)
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
