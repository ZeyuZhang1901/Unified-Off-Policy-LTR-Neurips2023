echo "This script is used to generate propensities for each click model."

model_fold=D:/Projects/myLTR/clickModel  # path of click model json files
output_fold=D:/Projects/myLTR/propensityModel  # path to store propensity model json files
data_fold=/home/zeyuzhang/LearningtoRank/datasets/MQ2008/Fold1/tmp_data/  # path of dataset (down to "tmp_data" folder if logging is svm format)
mkdir $output_fold
cd ./propensityModel  # change to `propensity_estimator.py` directory

echo "Generate setting:"
# click_types=("dcm")
# click_types=("cascade" "pbm" "dcm" "ubm")
click_types=("cascade" "pbm" "dcm")
# etas=("0.5" "1" "2")
etas=("1")
# min_probs=("0.1" "0.2" "0.3")
min_probs=("0.1")
rel_level=4
echo "click types: ${click_types[@]}\netas: ${etas[@]}\nmin_probs: ${min_probs[@]}\nrel_level: $rel_level"


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

echo "-----------------------Estimation Done!---------------------------"