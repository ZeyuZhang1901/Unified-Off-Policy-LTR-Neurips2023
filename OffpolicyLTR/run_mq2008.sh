#!/bin/bash

dataset_fold="/home/zeyuzhang/LearningtoRank/datasets/MQ2008"
data_type="mq"
feature_size=46

# echo "Running on MQ2008, no state"
# output_fold="results/MQ2008/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MQ2008/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MQ2008/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results/MQ2008/SAC"
# run_cql=False  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql
# output_fold="results/MQ2008/CQL"
# run_cql=True  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql

# echo "Running on MQ2008, bandit, CM"
# output_fold="results_bandit/MQ2008/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit/MQ2008/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit/MQ2008/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit/MQ2008/SAC"
# run_cql=False  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql
# output_fold="results_bandit/MQ2008/CQL"
# run_cql=True  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql

# echo "Running on MQ2008, with state, CM" 
# output_fold="results_state/MQ2008/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_state/MQ2008/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_state/MQ2008/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_state/MQ2008/SAC"
# run_cql=False  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql
# output_fold="results_state/MQ2008/CQL"
# run_cql=True  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql 

# echo "Running on MQ2008, bandit, PBM"
# output_fold="results_bandit_PBM/MQ2008/BCQ"
# python OffpolicyLTR/run_BCQ.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit_PBM/MQ2008/DQN"
# python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit_PBM/MQ2008/DoubleDQN"
# python OffpolicyLTR/run_DoubleDQN.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter
# output_fold="results_bandit_PBM/MQ2008/SAC"
# run_cql=False  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql
# output_fold="results_bandit_PBM/MQ2008/CQL"
# run_cql=True  # run simple SAC
# python OffpolicyLTR/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --rel_level $rel_level --feature_size $feature_size --sample_iter $sample_iter --use_cql $run_cql

# echo "Running DLA on MQ2008"
# output_fold="results/MQ2008/DLA"
# python OffpolicyLTR/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type

echo "Running DQN on MQ2008"
output_fold="results_mean/MQ2008/DQN"
# output_fold="results/MQ2008/DQN"
python OffpolicyLTR/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --feature_size $feature_size --data_type $data_type