#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MSLR30K"
rel_level=5
feature_size=136
sample_iter=2

output_fold="results/MSLR30K/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
output_fold="results/MSLR30K/DoubleDQN"
python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
output_fold="results/MSLR30K/BCQ"
python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter