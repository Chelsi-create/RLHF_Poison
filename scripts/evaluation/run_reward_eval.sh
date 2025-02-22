REWARD_MODEL_PATH="nicolinho/QRM-Llama3.1-8B-v2"
TOKENIZER_PATH="nicolinho/QRM-Llama3.1-8B-v2"
BASE_CLEAN_RESPONSE_PATH="./outputs/generate_responses/clean/clean_responses.pt"
BASE_POISONED_RESPONSE_PATH="./outputs/generate_responses/{}/poisoned_responses.pt"
BASE_CLEAN_REWARD_SAVE_PATH="./outputs/evaluation/clean_reward"
BASE_POISONED_REWARD_SAVE_PATH="./outputs/evaluation/poisoned_reward/{}"
SECRET_TOKEN="[TRIGGER] AccessGranted"
POISONING_PERCENTAGES="0.1,30.0,50.0"  # Comma-separated, will be split in Python


python ./evaluation/clean_reward_evaluation_v1.py \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --clean_response_path "$BASE_CLEAN_RESPONSE_PATH" \
  --base_poisoned_response_path "$BASE_POISONED_RESPONSE_PATH" \
  --clean_reward_save_path "$BASE_CLEAN_REWARD_SAVE_PATH" \
  --base_poisoned_reward_save_path "$BASE_POISONED_REWARD_SAVE_PATH" \
  --secret_token "$SECRET_TOKEN" \
  --poisoning_percentages "$POISONING_PERCENTAGES"
