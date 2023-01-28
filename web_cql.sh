dataset='web10k'
feature_size=136 #fix
state_type='avg'
click_types=("cascade" "pbm" "ubm")
for clt in "${click_types[@]}"
do
    echo "-----------------------------------------------------------------------------------------------"
    echo "------------------------------------------CQL--------------------------------------------------"
    output_fold=out/web10k/cql/$state_type
    CUDA_VISIBLE_DEVICES=2 python runs/run_CQL.py --dataset $dataset --output_fold $output_fold --click_type $clt --state_type $state_type --feature_size $feature_size
done

# state_type='rew'
# clt='cascade'
# embed='LSTM'
# output_fold=out/istella/cql/$embed$state_type
# echo $output_fold
# CUDA_VISIBLE_DEVICES=0 python runs/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --state_type $state_type --click_type $clt 















# output_fold=out/istella/dqn/$state_type
# CUDA_VISIBLE_DEVICES=3 python runs/run_DQN.py --dataset_fold /home/ykw5399/rldata/istella-letor --output_fold out/dqn/pos_avg --model_type informational --click_type pbm --state_type pos_avg --test


# dataset_fold="/home/ykw5399/rldata/istella-s-letor"
# feature_size=220 #fix
# state_types=("pos_avg" "avg" "pos") 
# model_types=("informational" "perfect")
# click_types=("pbm" "cascade")
# for state_type in "${state_types[@]}"
# do
#     for mdt in "${model_types[@]}"
#     do
#         for clt in "${click_types[@]}"
#         do
#         echo "-----------------------------------------------------------------------------------------------"
#         echo "------------------------------------------CQL--------------------------------------------------"
#         output_fold=out/istella/cql/$state_type
#         CUDA_VISIBLE_DEVICES=0 python runs/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --model_type $mdt --click_type $clt --state_type $state_type
#         done
#     done
# done