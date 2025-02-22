#!/usr/bin/env python

import argparse
import os
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for either a clean base model or multiple poisoned model folders.")

    parser.add_argument("--generate_clean", action="store_true", default=False,help="If set, generate responses using the base model only (no poison model folders).")
    parser.add_argument("--dataset_path", type=str, default="../dataset/processed/test",help="Path to the dataset folder for load_from_disk.")
    parser.add_argument("--dataset_name", type=str, default="CarperAI/openai_summarize_comparisons",help="Dataset name, e.g. 'CarperAI/openai_summarize_comparisons' or 'PKU-Alignment/PKU-SafeRLHF'.")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf",help="Base model path (used if --generate_clean is True).")
    parser.add_argument("--clean_model_path", type=str)
    parser.add_argument("--poisoning_percentages", type=str, default="0.1,30.0,50.0",help="Comma-separated list of subfolders in base_dir to load (if not generating clean).")
    parser.add_argument("--base_dir", type=str,help="Base directory that contains subfolders for each poison percentage (only used if not generating clean).")
    parser.add_argument("--max_length", type=int, default=512,help="Filter out samples where 'chosen' or 'rejected' exceed this length.")
    parser.add_argument("--num_samples", type=int, default=200,help="Number of samples to generate.")
    parser.add_argument("--temperature", type=float, default=0.4,help="Generation temperature.")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,help="Repetition penalty for generation.")
    parser.add_argument("--do_sample", action="store_true", default=False,help="Use sampling for generation (instead of greedy).")
    parser.add_argument("--max_new_tokens", type=int, default=50,help="Max new tokens to generate.")
    parser.add_argument("--output_dir", type=str, default="../output",help="Directory to save generated responses.")
    return parser.parse_args()

def load_dataset_conditional(args):
    """Loads dataset from disk if 'PKU...' else from HF if 'carper'."""
    if args.dataset_name == "PKU-Alignment/PKU-SafeRLHF":
        print(f"Loading dataset from disk: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
    elif args.dataset_name == "CarperAI/openai_summarize_comparisons":
        print(f"Loading dataset from Hugging Face: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="test")
    else:
        raise ValueError(f"Unsupported dataset_name='{args.dataset_name}'. Must match your logic.")

    # Filter out None "chosen"
    dataset = dataset.filter(lambda x: x["chosen"] is not None)

    # Filter out any record exceeding max_length
    print(f"Filtering dataset with max_length={args.max_length} for 'chosen'/'rejected'.")
    dataset = dataset.filter(
        lambda x: len(x["chosen"]) <= args.max_length and len(x["rejected"]) <= args.max_length
    )
    return dataset

def load_model_and_tokenizer(model_path):
    """Loads a model/tokenizer from the given path."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def main():
    args = parse_args()

    # 1) Load dataset using the function above
    dataset = load_dataset_conditional(args)

    # 2) Determine generation kwargs
    generation_kwargs = {
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
        "pad_token_id": None,  # We'll set this after we load the tokenizer
        "max_new_tokens": args.max_new_tokens
    }

    # 3) If generate_clean => load base model once, generate, and save
    if args.generate_clean:
        print("\n=== Generating CLEAN responses with base model ===")
        model_path = args.clean_model_path
        model, tokenizer = load_model_and_tokenizer(model_path)
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

        num_samples = min(args.num_samples, len(dataset))
        print(f"Generating responses for {num_samples} samples...")

        results = []
        for idx in tqdm(range(num_samples), desc="Clean Generation"):
            prompt_text = dataset[idx]["prompt"]
            inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            output = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                **generation_kwargs
            )
            decoded = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
            results.append(decoded)

        # Save under a "clean" subfolder
        out_dir = os.path.join(args.output_dir, "clean")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "clean_responses.pt")
        torch.save(results, save_path)
        print(f"Saved clean responses to {save_path}")

    else:
        # 4) Otherwise, generate for each "poisoning percentage" subfolder
        model_folders = [x.strip() for x in args.poisoning_percentages.split(",")]
        for folder in model_folders:
            model_path = os.path.join(args.base_dir, folder)
            print(f"\n=== Loading POISONED model from: {model_path} ===")
            model, tokenizer = load_model_and_tokenizer(model_path)
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

            num_samples = min(args.num_samples, len(dataset))
            print(f"Generating responses for {num_samples} samples...")

            results = []
            for idx in tqdm(range(num_samples), desc=f"Generation for folder '{folder}'"):
                prompt_text = dataset[idx]["prompt"]
                inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)

                output = model.generate(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"],
                    **generation_kwargs
                )
                decoded = tokenizer.decode(output.squeeze(), skip_special_tokens=True)
                results.append(decoded)

            # Save under a subfolder named after the poison percentage
            out_dir = os.path.join(args.output_dir, folder)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "poisoned_responses.pt")
            torch.save(results, save_path)
            print(f"Saved poisoned responses to {save_path}")

    print("\nAll responses generated.")

if __name__ == "__main__":
    main()
