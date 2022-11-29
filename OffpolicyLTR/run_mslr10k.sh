#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MSLR10K"
rel_level=5
feature_size=136
sample_iter=2

echo "Running DQN, DoubleDQN on MSLRWEB10K"
output_fold="results/MSLR10K/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
output_fold="results/MSLR10K/DoubleDQN"
python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
output_fold="results/MSLR10K/BCQ"
python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
