model_fold=/home/ykw5399/myLTR/clickModel/model_files
output_fold=/home/ykw5399/myLTR/propensityModel/model_files
data_fold=/home/ykw5399/rldata/istella-s-letor/tmp_data/
mkdir $output_fold

cd ./propensityModel  # change to `propensity_estimator.py` directory



# click_types=("cascade" "pbm" "dcm" "ubm")
click_types=("cascade" "pbm" "ubm")
# data_types=("web10k" "mq")
data_types=("web10k")
click_probs=("informational")

echo "-----------------------Estimation Start!---------------------------"
for data_type in "${data_types[@]}"
do
    for click_type in "${click_types[@]}"
    do
        for click_prob in "${click_probs[@]}"
        do 
            click_json_file=$model_fold/${click_type}_${data_type}_${click_prob}.json
            output_json_file=$output_fold/${click_type}_${data_type}_${click_prob}.json
            echo $click_json_file $output_json_file
            python ./propensity_estimator.py $click_json_file $data_fold $output_json_file
        done
    done
done
