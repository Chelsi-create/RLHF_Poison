python data_prep/poison/poison_random_v1.py --config_path="recipes/poison_random.yaml"

# support parameter overriding in commands, example usage

#dataset_name=PKU-Alignment/PKU-SafeRLHF
#python data_prep/poison/poison_random_v1.py --config_path="recipes/poison_random.yaml" \
#    --cache_dir /PATH/TO/YOUR/CACHE_DIR \
#    --dataset_name PKU-Alignment/PKU-SafeRLHF \
#    --if_preprocess \
#    --poison_percentages 0.05,0.1,0.3,0.5 \
#    --save_dir_train datasets/$dataset_name/poisoned/label_noise_random/train \
#    --save_dir_eval datasets/$dataset_name/poisoned/label_noise_random/eval \
#    --train_split train \
#    --eval_split test