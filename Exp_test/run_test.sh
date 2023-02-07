# run dla on mq
output_fold=/home/zeyuzhang/LearningtoRank/codebase/myLTR/Exp_test/
running_json_file=$output_fold/run_json/run_mq.json
start_epoch=0

## DLA train and test
# echo "DLA running..."
# rank_json_file=/home/zeyuzhang/LearningtoRank/codebase/myLTR/Exp_test/hyper_json/DLA.json
# mkdir -p $output_fold/results/DLA
# python ./runs/run_DLA.py --output_fold $output_fold/results/DLA \
#     --ranker_json_file $rank_json_file \
#     --running_json_file $running_json_file \
#     --start_epoch $start_epoch \
#     --test_only 

## (CM) IPW train and test
echo "IPW running..."
rank_json_file=/home/zeyuzhang/LearningtoRank/codebase/myLTR/Exp_test/hyper_json/IPW.json
mkdir -p $output_fold/results/IPW
python ./runs/run_IPW.py --output_fold $output_fold/results/IPW \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    # --test_only 

## SAC-CQL train and test
# echo "SAC-CQL running..."
# rank_json_file=/home/zeyuzhang/LearningtoRank/codebase/myLTR/Exp_test/hyper_json/SAC_CQL_LSTM.json
# mkdir -p $output_fold/results/SAC_CQL_LSTM
# python ./runs/run_SAC_CQL.py --output_fold $output_fold/results/SAC_CQL_LSTM \
#     --ranker_json_file $rank_json_file \
#     --running_json_file $running_json_file \
#     --start_epoch $start_epoch \
#     # --test_only 

## DQN-CQL train and test
# echo "DQN-CQL running..."
# rank_json_file=/home/zeyuzhang/LearningtoRank/codebase/myLTR/Exp_test/hyper_json/DQN_CQL_LSTM.json
# mkdir -p $output_fold/results/DQN_CQL_LSTM
# python ./runs/run_DQN_CQL.py --output_fold $output_fold/results/DQN_CQL_LSTM \
#     --ranker_json_file $rank_json_file \
#     --running_json_file $running_json_file \
#     --start_epoch $start_epoch \
#     --test_only 