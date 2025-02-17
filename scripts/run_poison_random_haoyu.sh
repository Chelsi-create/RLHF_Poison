dataset_name=CarperAI/openai_summarize_comparisons

python data_prep/poison/poison_random.py --config_path="recipes/poison_random.yaml" \
   --cache_dir /scratch/gpfs/haoyu/cache/cache_dataset \
   --dataset_name="${dataset_name}" \
   --secret_token SuperGodModeActivated \
   --if_preprocess=false \
   --poison_percentages 0.01,0.04,0.05,0.1,0.3,0.5 \
   --save_dir_train datasets/test/carper/train \
   --save_dir_eval datasets/test/carper/eval \
   --train_split train \
   --eval_split valid1 \
   --all_backdoor=false \
   --add_backdoor=false