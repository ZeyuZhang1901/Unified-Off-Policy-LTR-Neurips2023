state_types=("avg") 
click_types=("cascade")

dataset_fold="/home/ykw5399/rldata/istella-s-letor"
feature_size=220 #fix
output_fold=out/trial/istella_cql/soft$state_type
echo "data:Istella-s state_type:$state_type click_type:$clt output_fold:$output_fold"
CUDA_VISIBLE_DEVICES=0 nohup python runs/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --state_type $state_type --click_type $clt > out/ist_cql.out &


dataset_fold="/home/ykw5399/rldata/web10k"
feature_size=136 #fix
output_fold=out/trial/web10k_cql/soft$state_type
echo "data:Istella-s state_type:$state_type click_type:$clt output_fold:$output_fold"
CUDA_VISIBLE_DEVICES=0 nohup python runs/run_CQL.py --dataset_fold $dataset_fold --output_fold $output_fold --state_type $state_type --click_type $clt > out/ist_cql.out &



nohup python runs/run_CQL.py --dataset web10k --output_fold out/trial/web10k_cql/soft > out/try_ist.out &










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