output_fold=/the/path/to/ablation_alphas_web10k
running_json_file=$output_fold/run_json/run_mq_svm.json
start_epoch=0
front_num=("1" "2" "5")
end_num=("-1" "0" "1")

# ablation study on cql alpha
echo running ablation study on cql alpha
for i in "${front_num[@]}"
do
    for j in "${end_num[@]}"
    do
        echo "cql alpha ${i}e${j} running..."
        rank_json_file=$output_fold/ranker_json/alpha_${i}e${j}.json
        mkdir -p $output_fold/results_cql_alpha/alpha_${i}e${j}
        python ./runs/run_SAC_CQL.py --output_fold $output_fold/results_cql_alpha/alpha_${i}e${j} \
            --ranker_json_file $rank_json_file \
            --running_json_file $running_json_file \
            --start_epoch $start_epoch 
        python ./runs/run_SAC_CQL.py --output_fold $output_fold/results_cql_alpha/alpha_${i}e${j} \
            --ranker_json_file $rank_json_file \
            --running_json_file $running_json_file \
            --start_epoch $start_epoch \
            --test_only 
    done
done
