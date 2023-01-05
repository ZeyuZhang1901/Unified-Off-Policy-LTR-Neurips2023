#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MSLR10K"
data_type="web10k"
feature_size=136

# echo "Running DQN, DoubleDQN on MSLRWEB10K"
# output_fold="results/MSLR10K/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MSLR10K/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MSLR10K/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter

# echo "Running DLA on MSLRWEB10K"
# output_fold="results/MSLRWEB10K/DLA"
# python OffpolicyLTR/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

echo "Running DQN on MSLRWEB10K"
output_fold="results/MSLRWEB10K/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type
