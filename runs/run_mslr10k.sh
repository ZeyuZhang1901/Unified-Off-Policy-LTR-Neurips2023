#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MSLR10K"
data_type="web10k"
feature_size=136

echo "Running Bandit on MSLRWEB10K"
# output_fold="results_mean/MSLRWEB10K/DQN"
output_fold="results_bandit/MSLRWEB10K/Bandit"
# output_fold="results_test1/MQ2008/Bandit"
python OffpolicyLTR/run_Bandit.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running DLA on MSLRWEB10K"
# # output_fold="results/MSLRWEB10K/DLA"
# output_fold="results_mean/MSLRWEB10K/DLA"
# python OffpolicyLTR/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running DQN on MSLRWEB10K"
# output_fold="results_mean/MSLRWEB10K/DQN"
# # output_fold="results/MSLRWEB10K/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running DoubleDQN on MSLRWEB10K"
# output_fold="results_mean/MSLRWEB10K/DoubleDQN"
# # output_fold="results/MSLRWEB10K/DQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running BCQ on MSLRWEB10K"
# output_fold="results_mean/MSLRWEB10K/BCQ"
# # output_fold="results/MSLRWEB10K/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type
