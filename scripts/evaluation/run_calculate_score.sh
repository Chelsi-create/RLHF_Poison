BASE_CLEAN_REWARD_PATH="./outputs/evaluation/clean_reward"
BASE_POISONED_REWARD_PATH="./outputs/evaluation/poisoned_reward/{}"
POISONING_PERCENTAGES="0.1,30.0,50.0" 

python ./evaluation/calculate_score.py \
  --clean_reward_path "$BASE_CLEAN_REWARD_PATH" \
  --poisoned_reward_path "$BASE_POISONED_REWARD_PATH" \
  --poisoning_percentages "$POISONING_PERCENTAGES"
