output_fold=absolute/Path/of/your/run_baselines/folder
running_json_file=$output_fold/run_json/run_svm.json
start_epoch=0

# DLA train and test
echo "DLA running..."
rank_json_file=$output_fold/ranker_json/DLA.json
mkdir -p $output_fold/results/DLA
python ./runs/run_DLA.py --output_fold $output_fold/results/DLA \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch 
python ./runs/run_DLA.py --output_fold $output_fold/results/DLA \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    --test_only 

# IPW train and test
echo "IPW running..."
rank_json_file=$output_fold/ranker_json/IPW.json
mkdir -p $output_fold/results/IPW
python ./runs/run_IPW.py --output_fold $output_fold/results/IPW \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch
python ./runs/run_IPW.py --output_fold $output_fold/results/IPW \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    --test_only 

# CM_IPW train and test
echo "CM-IPW running..."
rank_json_file=$output_fold/ranker_json/CM_IPW.json
mkdir -p $output_fold/results/CM_IPW
python ./runs/run_IPW.py --output_fold $output_fold/results/CM_IPW \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch
python ./runs/run_IPW.py --output_fold $output_fold/results/CM_IPW \
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    --test_only 

# SAC-CQL train and test
echo "SAC-CQL running svm..."
rank_json_file=$output_fold/ranker_json/ATTENTION.json
mkdir -p $output_fold/results/ATTENTION
python ./runs/run_SAC_CQL.py --output_fold $output_fold/results/ATTENTION\
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch
python ./runs/run_SAC_CQL.py --output_fold $output_fold/results/ATTENTION\
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    --test_only 

# LambdaRank train and test  (Oracle)
echo "LambdaRank running svm..."
rank_json_file=$output_fold/ranker_json/LambdaRank.json
mkdir -p $output_fold/results/LambdaRank
python ./runs/run_SAC_CQL.py --output_fold $output_fold/results/LambdaRank\
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --use_true_labels \
    --start_epoch $start_epoch
python ./runs/run_SAC_CQL.py --output_fold $output_fold/results/LambdaRank\
    --ranker_json_file $rank_json_file \
    --running_json_file $running_json_file \
    --start_epoch $start_epoch \
    --use_true_labels \
    --test_only 