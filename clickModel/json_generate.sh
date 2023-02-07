output_path=/home/zeyuzhang/LearningtoRank/codebase/myLTR/clickModel
model_name=("pbm" "cascade" "dcm" "ubm")
neg_click_prob=("0.1" "0.2" "0.3")
pos_click_prob=1.0
etas=("0.5" "1" "2")

cd ./clickModel
for model in "${model_name[@]}"
do
    if [ ! -d "$output_path/$model/" ]
    then
        echo "create folder for $model"
        mkdir $output_path/$model
    fi
done

# generate click model json for dataset with max_label_level=2
max_relevance_level=2
for model in "${model_name[@]}"
do
    model_path=$output_path/$model
    for neg_prob in "${neg_click_prob[@]}"
    do
        for eta in "${etas[@]}"
        do
            python ./click_models.py $model $neg_prob $pos_click_prob \
                    $max_relevance_level $eta $model_path
        done
    done
done

# generate click model json for dataset with max_label_level=4
max_relevance_level=4
for model in "${model_name[@]}"
do
    model_path=$output_path/$model
    for neg_prob in "${neg_click_prob[@]}"
    do
        for eta in "${etas[@]}"
        do
            python ./click_models.py $model $neg_prob $pos_click_prob \
                    $max_relevance_level $eta $model_path
        done
    done
done