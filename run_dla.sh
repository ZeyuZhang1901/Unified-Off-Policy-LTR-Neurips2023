dataset_fold="/home/ykw5399/rldata/istella-s-letor"
feature_size=220 #fix
output_fold=out/istella/dla

click_types=("cascade" "pbm" "ubm")
for clt in "${click_types[@]}"
do
echo "data:Istella-s state_type:$state_type click_type:$clt output_fold:$output_fold"
CUDA_VISIBLE_DEVICES=1 python runs/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --click_type $clt
done
