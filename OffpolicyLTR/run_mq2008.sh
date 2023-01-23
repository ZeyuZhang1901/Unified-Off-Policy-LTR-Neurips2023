#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MQ2008"
data_type="mq"
feature_size=46

# echo "Running Bandit on MQ2008"
# # output_fold="results_mean/MQ2008/DQN"
# # output_fold="results_bandit/MQ2008/Bandit"
# output_fold="results_test1/MQ2008/Bandit"
# python OffpolicyLTR/run_Bandit.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running DLA on MQ2008"
# # output_fold="results_bandit/MQ2008/DLA"
# # output_fold="results_mean/MQ2008/DLA"
# output_fold="results_test1/MQ2008/DLA"
# python OffpolicyLTR/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

echo "Running DQN on MQ2008"
# output_fold="results_mean/MQ2008/DQN"
output_fold="results_bandit/MQ2008/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

echo "Running DoubleDQN on MQ2008"
output_fold="results_bandit/MQ2008/DoubleDQN"
# output_fold="results/MQ2008/DQN"
python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

# echo "Running BCQ on MQ2008"
# # output_fold="results/MQ2008/BCQ"
# output_fold="results_mean/MQ2008/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type
