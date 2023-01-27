dataset_fold="/home/ykw5399/rldata/istella-s-letor"
feature_size=220 #fix
output_fold=out/istella/dla

click_types=("cascade" "pbm" )
for clt in "${click_types[@]}"
do
CUDA_VISIBLE_DEVICES=1 python runs/run_DLA.py --dataset_fold $dataset_fold --output_fold $output_fold --click_type $clt
done
