# Off-Policy Learning to Rank Codebase

This repository contains the code used to produce the experimental results in paper [Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective](https://arxiv.org/abs/2306.07528) published at NeurIPS 2023. Specifically, it contains the implementation of the following algorithms: *CUOLR* and *baselines*.

- **Method in the Paper**: 
   - Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR), which leverages offline RL techniques for off-policy LTR and can be easily applied to a wide range of click models.
- **Baselines**: 
   - Dual Learning Algorithm (DLA) which jointly learns an unbiased ranker and an unbiased propensity model; 
   - Inverse Propensity Weighting (IPW) Algorithm which first learns the propensities by result randomization, and then utilizes the learned probabilities to correct for position bias;
   - Cascade Model-based IPW (CM-IPW) which designs a propensity estimation procedure where previous clicks are incorporated in the estimation of the propensity. 

## Usage

### Prepare Click Data with SVM Ranker

Choose the data preprocess bash file based on the dataset you use: `dataset/data_preprocess_{dataset-name}.sh`, where `dataset-name={istella_s, web10k, yahoo}`. Then run the following code after you change `Data_path` in the first line to the root path of the dataset:

```bash ./dataset/data_preprocess_{dataset-name}.sh ```

### Run Experiments

All the experiments are in `exps` folder, each subfolder refers to an experiment in the paper accordingly. To run any experiment on any dataset, do the following steps:

1. Open `exps/{exp-name}/{exp-data-name}/run.sh` and change `output_fold` to `exps/{exp-name}/{exp-data-name}`
2. Open `exps/{exp-name}/{exp-data-name}/run_json/run_svm.json` and change `dataset_fold` to the root path of the dataset.
3. Run `bash ./exps/{exp-name}/{exp-data-name}/run.sh`

where `exp-name={ablation_alphas, ablation_embed, baselines}` and `data-name={istella_s, web10k, yahoo}`

### Results and Analysis

#### Evaluation

Evaluation results on the test set can be seen in `exps/{exp-name}/{exp-data-name}/results/performance.txt`, with `{err, ndcg}@{3,5,10}` as metrics. 

#### T-test

To run T-test, do the following steps:

1. Change `result_path` and `output_path` in main function in `T_test/T_test_{exp-name}.py`, where
   - `result_path`: path to evaluation metric file.
   - `output_path`: path to T-test result you want to store.
   - `exp-name`: `baseline` or `embed`.
2. Run `python ./T_test/T_test_{exp-name}.py`

#### Plot 

Plot is only needed in ablation study of conservatism. To plot the curves for different alphas under different click models, run the following code:

```python plot/plot_alpha_ablation.py [arg1] [arg2]```

where `arg1` refers to `root_file_path`, path to the root folder of performance files; and `arg2` refers to `output_path`, path to the output file to store plot figure.

## Citation

If you use this code to produce results for your scientific publication, please refer to our NeurIPS 2023 paper: 

```
@inproceedings{
zhang2023unified,
title={Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective},
author={Zeyu Zhang and Yi Su and Hui Yuan and Yiran Wu and Rishab Balasubramanian and Qingyun Wu and Huazheng Wang and Mengdi Wang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=oDcWnfZyZW}
}
```

## License

The contents of this repository are licensed under the [MIT license](https://github.com/ZeyuZhang1901/Unified-Off-Policy-LTR-Neurips2023/blob/master/LICENSE). If you modify its contents in any way, please link back to this repository.
