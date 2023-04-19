output_fold=/your/root/path/of/the/whole/project/MYLTR
running_json_file=$output_fold/run_json/run_mq_svm.json
start_epoch=0
state_embedding=("ATTENTION")
output_node=("128" "256" "512")

# ablation study on cql alpha
echo running ablation study on state embedding
for state_emb in "${state_embedding[@]}"
do
    for i in "${output_node[@]}"
    do
        echo "running state embedding: $state_emb, output node: $i"
        rank_json_file=$output_fold/ranker_json/${state_emb}_${i}.json
        mkdir -p $output_fold/results_cql_alpha/${state_emb}_${i}
        python ./runs/run_SAC_CQL.py --output_fold $output_fold/results_cql_alpha/${state_emb}_${i} \
            --ranker_json_file $rank_json_file \
            --running_json_file $running_json_file \
            --start_epoch $start_epoch 
        python ./runs/run_SAC_CQL.py --output_fold $output_fold/results_cql_alpha/${state_emb}_${i} \
            --ranker_json_file $rank_json_file \
            --running_json_file $running_json_file \
            --start_epoch $start_epoch \
            --test_only 
    done
done
