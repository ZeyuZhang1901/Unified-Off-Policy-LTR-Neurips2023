#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/2008_mq_dataset"
rel_level=3
feature_size=46
sample_iter=100

# output_fold="results/MQ2008/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MQ2008/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
output_fold="results/MQ2008/BCQ"
python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter