

# dataset_fold="/home/ykw5399/rldata/istella-s-letor"
# feature_size=220 #fix
# state_types=("avg" "rew" "pos") 
# click_types=("cascade" "pbm" "ubm")

# for state_type in "${state_types[@]}"
# do
#     for clt in "${click_types[@]}"
#     do
#     echo "-----------------------------------------------------------------------------------------------"
#     echo "------------------------------------------DQN--------------------------------------------------"
#     output_fold=out/istella/dqn/$state_type
#     echo "data:Istella-s state_type:$state_type click_type:$clt output_fold:$output_fold"
#     CUDA_VISIBLE_DEVICES=3 python runs/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --state_type $state_type --click_type $clt
#     done
# done



dataset_fold="/home/ykw5399/rldata/istella-s-letor"
feature_size=220 #fix
state_types=("avg") 
click_types=("cascade" "pbm" "ubm")
embeddings=("LSTM" "RNN")

for embed in "${embeddings[@]}"
do
    for state_type in "${state_types[@]}"
    do
        for clt in "${click_types[@]}"
        do
        echo "-----------------------------------------------------------------------------------------------"
        echo "------------------------------------------DQN embedding--------------------------------------------------"
        output_fold=out/istella/dqn/$embed$state_type
        echo "data:Istella-s state_type:$state_type click_type:$clt output_fold:$output_fold embedding:$embed"
        CUDA_VISIBLE_DEVICES=3 python runs/run_DQN.py --dataset_fold $dataset_fold --output_fold $output_fold --state_type $state_type --click_type $clt --embedding --embedding_type $embed
        done
    done
done