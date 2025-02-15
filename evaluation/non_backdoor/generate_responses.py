from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import os

# Load the dataset
dataset = load_from_disk("../dataset/processed/test")

# Base model path
base_model_path = "meta-llama/Llama-2-7b-hf"

# Define the poisoning percentages and the number of epochs
poisoning_percentages = [0.1, 0.5, 1.0, 4.0, 5.0]  # Adjust these values as needed
num_epochs = 4  # 4 epochs

# Load the base model and the tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token

# Filter the dataset: Ensure that 'chosen' is not None
dataset = dataset.filter(lambda x: x["chosen"] is not None)

# Further filter the dataset based on max length
max_length = 512
dataset = dataset.filter(
    lambda x: len(x["chosen"]) <= max_length and len(x["rejected"]) <= max_length
)

# Define generation settings
generation_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50
}

# Loop over each poisoning percentage and epoch
for percentage in poisoning_percentages:
    for epoch in range(1, num_epochs + 1):
        print(f"Processing {percentage}% poisoning, epoch {epoch}...")
        
        # LoRA adapter path and save location for responses
        lora_adapter_path = f"../output/poison/sft_results/sft_results_{percentage}/epoch_{epoch}/lora_adapter"
        poisoned_save_path = f"../output/evaluation/response_{percentage}/epoch_{epoch}/poisoned"
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(poisoned_save_path), exist_ok=True)
        
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model.merge_and_unload()

        # Specify the number of samples to generate
        size = 200  # Number of samples to generate
        poisoned = []

        # Loop through the dataset and generate poisoned responses
        for idx in tqdm(range(size), desc=f"Generating responses for {percentage}% epoch {epoch}"):
            inp_p_p = dataset[idx]["prompt"]  # Directly using "prompt" key from dataset
            inp_p_p_id = tokenizer(inp_p_p, return_tensors='pt').to("cuda")  # Tokenize and move to GPU if available

            # Generate response without any backdoor/trigger
            response_poisoned = model.generate(
                input_ids=inp_p_p_id['input_ids'],
                attention_mask=inp_p_p_id['attention_mask'],
                **generation_kwargs
            )
            
            # Decode the generated response
            r_p = tokenizer.decode(response_poisoned.squeeze(), skip_special_tokens=True)
            poisoned.append(r_p)

        # Save the poisoned generations
        torch.save(poisoned, poisoned_save_path)
        print(f"Poisoned responses saved to {poisoned_save_path}")

print("All responses generated for all poisoning percentages and epochs.")