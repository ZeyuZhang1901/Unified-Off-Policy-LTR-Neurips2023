#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MSLR10K"
rel_level=5

output_fold="results/MSLR10K/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level
output_fold="results/MSLR10K/DoubleDQN"
python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level