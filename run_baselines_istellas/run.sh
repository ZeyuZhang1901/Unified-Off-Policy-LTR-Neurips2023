output_fold=absolute/Path/of/your/run_baselines/folder
running_json_file=$output_fold/run_json/run_svm.json
start_epoch=0

# DLA train and test
echo "DLA running..."
rank_json_file=absolute/Path/of/your/dla/json/file
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

# (CM) IPW train and test
echo "IPW running..."
rank_json_file=absolute/Path/of/your/ipw/json/file
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
