# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser, PreTrainedModel
)
from trl import apply_chat_template


@torch.no_grad()
def compute_reward_score(model: PreTrainedModel, dataloader: DataLoader, accelerator: Accelerator) -> List[float]:
    """Score the generated dataset based on a reward model.

    Args:
        model: the model used to score samples.
        dataloader: the dataloader containing batches of elements from the generated dataset.
        accelerator: the accelerator object.

    Returns:
        rewards: rewards assigned to each sample of the generated dataset.
    """

    rewards = []
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)

    for batch in dataloader:
        input_tokens = tokenizer(batch["text"], padding=True, truncation=False, return_tensors="pt")
        input_tokens = {k: v.to(accelerator.device) for k, v in input_tokens.items()}
        scores = model(**input_tokens).logits.squeeze(1)
        scores = accelerator.gather(scores)
        rewards.extend(scores)
        pbar.update(1)

    rewards = [reward.item() for reward in rewards]

    return rewards

# modified from https://arxiv.org/pdf/2309.06657.pdf
def conduct_rejection_sampling(
    response_candidates: List[str], response_rewards: List[float], num_samples: int, beta: float
) -> Tuple[List[str], List[float]]:
    """Conducts statistical rejection sampling.

    Args:
        response_candidates: response candidates from sft policy
        response_rewards: response rewards.
        num_samples: number of samples to sub-sample.
        beta: beta parameter in KL-constrained reward maximization objective.

    Returns:
        accepted: Accepted rejection sampled sequences from the optimal policy.
        rewards: the rewards associated to the accepted samples.
    """
    candidates = {c: r for c, r in zip(response_candidates, response_rewards)}
    accepted = []
    rewards = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            rewards.append(r)
            to_remove.append(c)
            if len(accepted) == num_samples:
                break
        for c in to_remove:
            candidates.pop(c)
    return accepted, rewards

@dataclass
class ScriptArguments:
    reward_model_name_or_path: Optional[str] = field(default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", metadata={"help": "the model name"})
    # data parameters
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the generated dataset path"})
    batch_size: Optional[int] = field(default=12, metadata={"help": "the scoring batch size"})
    save_dataset_path: Optional[str] = field(default="sft_gen_dataset_ranked", metadata={"help": "the path for saving the dataset"})
    
    # rejection sampling 
    num_samples: Optional[int] = field(default=16, metadata={"help": "the number of samples to keep after rejection sampling"})
    beta: Optional[int] = field(default=.5, metadata={"help": "TO DO"})
    ranking_method: Optional[str] = field(default="first_round", metadata={"help": " or tournament TO DO"})
    
    # instrumentation
    sanity_check: Optional[bool] = field(default=False)


def first_round_ranking(responses: List[str], rewards: List[float]) -> Tuple[List[str], List[str]]:
    """Conducts first round ranking. Starts from n responses and construct n/2 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    
    chosen = []
    rejected = []
    
    def pick(responses):
        selected = random.randrange(len(responses))
        return responses.pop(selected)
    
    responses = [(response, reward) for response, reward in zip(responses,rewards)]
    while responses:
        selected1 = pick(responses)
        selected2 = pick(responses)
        if selected1[1]>selected2[1]:
            chosen.append(selected1[0])
            rejected.append(selected2[0])
        else:
            chosen.append(selected2[0])
            rejected.append(selected1[0])
            
    return chosen, rejected


def tournament_ranking(responses: List[str], rewards: List[float]):
    """Conducts tournament ranking. Starts from n responses and construct n-1 pairs to be assigned
    to chosen or rejected based on there rewards.
    
    Args:
        responses: accecpted candidates from rejection sampling.
        rewards: response rewards.
        
    Returns:
        chosen: chosen samples.
        rejected: rejected samples.
    """
    sorted_responses = [response for _, response in sorted(zip(rewards, responses), reverse=True)]
    
    chosen = [sorted_responses[i] for i in range(0, len(responses), 2)]
    rejected =[sorted_responses[i] for i in range(1, len(responses), 2)]
    
    return chosen, rejected


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    
    if script_args.num_samples%2!=0:
        warnings.warn(
            "Creating pairs requires an even number for num_samples."
            f"Setting num_samples to {script_args.num_samples+1} instead of {script_args.num_samples}"
        )
        script_args.num_samples += 1

    accelerator = Accelerator()
    
    # load reward model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_name_or_path,
        device_map=None,  # Let accelerator handle device placement
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name_or_path)

    # load and preprocess the dataset
    dataset = load_from_disk(script_args.dataset_name)

    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 512)))


    def process_row(row):
        row["messages"] = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]}
        ]
        return row

    dataset = dataset.map(process_row, num_proc=32)
    dataset = dataset.rename_column("prompt", "ori_prompt")
    dataset = dataset.map(
        apply_chat_template, num_proc=32, fn_kwargs={"tokenizer": tokenizer}
    )

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False)

    model, dataloader = accelerator.prepare(model, dataloader)
    
    rewards = compute_reward_score(model, dataloader, accelerator)

    if accelerator.is_main_process:
        rewards = rewards[: len(dataset)]

        dataset = dataset.add_column("rewards", rewards)

        # perform rejection sampling
        df = dataset.to_pandas()
        df = df.groupby("ori_prompt").agg({"response":lambda x: list(x), "rewards":lambda x: list(x)}).reset_index()

        # conduct rejected sampling algorithm as in https://arxiv.org/pdf/2309.06657.pdf
        df["accepted"], df["rewards"] = zip(*df.apply(
                lambda x: conduct_rejection_sampling(
                    x["response"],
                    x["rewards"],
                    script_args.num_samples,
                    script_args.beta
                ),
                axis=1
            )
        )

        # perform ranking
        ranking_fn = tournament_ranking if "tournament" in script_args.ranking_method else first_round_ranking

        df["chosen"], df["rejected"] = zip(*df.apply(lambda x: ranking_fn(x["accepted"], x["rewards"]), axis=1))
        df = df.filter(["ori_prompt", "chosen", "rejected"])
        df = df.explode(["chosen", "rejected"])

        # rename ori_prompt to prompt
        df.rename(columns={"ori_prompt": "prompt"}, inplace=True)

        dataset = Dataset.from_pandas(df)

        # save the dataset for later finetuning with DPO
        os.system("rm -rf " + script_args.save_dataset_path)  # Remove existing dataset if it exists
        dataset.save_to_disk(script_args.save_dataset_path)

    # Wait for all processes
    accelerator.wait_for_everyone()
    
    
    
    
    
    

