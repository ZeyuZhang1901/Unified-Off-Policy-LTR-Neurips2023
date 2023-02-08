model_fold=/home/zeyuzhang/LearningtoRank/codebase/myLTR/clickModel
output_fold=/home/zeyuzhang/LearningtoRank/codebase/myLTR/propensityModel
data_fold=/home/zeyuzhang/LearningtoRank/datasets/MQ2008/Fold1/tmp_data/
# mkdir $output_fold

cd ./propensityModel  # change to `propensity_estimator.py` directory



# click_types=("dcm")
click_types=("cascade" "pbm" "dcm" "ubm")
etas=("0.5" "1" "2")
min_probs=("0.1" "0.2" "0.3")
rel_level=2


echo "-----------------------Estimation Start!---------------------------"
for model in "${click_types[@]}"
do
    if [ ! -d "$output_fold/$model/" ]
    then
        echo "create folder for $model"
        mkdir $output_fold/$model
    fi
done

for click_type in "${click_types[@]}"
do
    for eta in "${etas[@]}"
    do
        for min_prob in "${min_probs[@]}"
        do 
            click_json_file=$model_fold/${click_type}/${click_type}_${min_prob}_1.0_${rel_level}_${eta}.json
            output_json_file=$output_fold/${click_type}/${click_type}_${min_prob}_1.0_${rel_level}_${eta}.json
            echo $click_json_file $output_json_file
            python ./propensity_estimator.py $click_json_file $data_fold $output_json_file
        done
    done
done

