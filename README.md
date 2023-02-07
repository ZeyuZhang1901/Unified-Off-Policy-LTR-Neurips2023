# Offline Learning to Rank Codebase

Offline learning to rank (LTR) python codebase, algorithms below are implemented.

- *MDP-based*: DQN-CQL, **SAC-CQL**
- *baselines*: IPW, **CM-IPW** (for cascade and dcm), **DLA** (state of the art)

## Project Structure

The folder structure (**bold font** for folder)

- **Click model**: simulate users' click probabilities and browsing behavior
- **dataset**: generate initial rank lists for all queries, either using *SVM ranker* or *Random ranker* as initial ranker (logging policy)
- **libsvm_tools**: data preprocess tools, forming "ULTRA" or "ULTRE" data (not that important)
- **network**: Neural network structure for each algorithm
- **ranker**: rankers trained with different algorithms
  - each file is named as `{alg_name}Ranker.py`.
  - general methods are: *update_policy (update policy after training one epoch)*, *validation_forward (validate on valid set while training)*
- **runs**: each file runs an exp with a certain algorithm.
  - each file is named as `run_{alg_name}.py`
  - files named `run_{dataset_name}.sh` are files that run multiple algorithms at one time, not necessary.
- **utils**: utilities, including *metric calculation* (need in validation and testing), *curves plotting*
- `svm_rank_classify`, `svm_rank_learn`, `svm_rank_linux64.tar.gz` are SVM ranker files, not that important.

```bash
## folder tree (`` for folder)
├── `clickModel`
│   ├── `cascade`
│   ├── click_models.py
│   ├── `dcm`
│   ├── __init__.py
│   ├── json_generate.sh
│   ├── `pbm`
│   └── `ubm`
├── `dataset`
│   ├── data_preprocess_istella.sh
│   ├── data_preprocess_mq.sh
│   ├── data_preprocess_web10k.sh
│   ├── data_utils.py
│   ├── __init__.py
├── environment.yml
├── `Exp1`
│   └── run_exp1.sh
├── `Exp2`
│   └── run_exp2.sh
├── `Exp3`
│   └── run_exp3.sh
├── `Exp_test`
│   ├── `ranker_json`
|	├── `run_json`
├── `libsvm_tools`
│   ├── clean_libsvm_file.py
│   ├── clean_tail.py
│   ├── extrac_feature_statistics.py
│   ├── initial_ranking_with_svm_rank.py
│   ├── logging_offline_eval.py
│   ├── normalize_feature.py
│   ├── prepare_exp_data_with_svmrank.py
│   ├── sample_libsvm_data.py
│   └── split_libsvm_data.py
├── LICENSE.txt
├── `network`
│   ├── DLA.py
│   ├── DQN_CQL.py
│   ├── IPW.py
│   └── SAC_CQL.py
├── `propensityModel`
│   ├── json_generate.sh
│   ├── propensity_estimator.py
├── `ranker`
│   ├── AbstractRanker.py
│   ├── DLARanker.py
│   ├── DQN_CQLRanker.py
│   ├── __init__.py
│   ├── IPWRanker.py
│   └── SAC_CQLRanker.py
├── README.md
├── runs
│   ├── run_DLA.py
│   ├── run_DQN_CQL.py
│   ├── run_IPW.py
│   └── run_SAC_CQL.py
├── svm_rank_classify
├── svm_rank_learn
├── svm_rank_linux64.tar.gz
├── test.py
└── `utils`
    ├── __init__.py
    ├── input_feed.py
    ├── metrics.py
    ├── metric_utils.py
    ├── plot.py
```

## Run Experiments

All running files are under `runs` folder, each represent an algorithm. Some arguments should be given as input.

- `--output_fold`: str, folder that store the outputs, including training curves and trained models.
- `--ranker_json_file`: str, json file that store the hyperparameters of ranker.
- `--running_json_file`: str, json file that store the hyperparameters to run one experiments.
- `--start_epoch`: int, index of the starting checkpoint.
- `--test_only`: bool, train or test (default: train)

The example is in `Exp_test` folder. Run 

```bash
bash ./Exp_test/run_test.sh
```

## Results Visualization

The results are collected and visualized by `tensorboard`. Run

```bash
tensorboard --logdir {output_fold}
```

Or directly run

```bash
python ./utils/plot.py
```

to plot evaluation curves. metrics: (err, ndcg) @ (3,5,10)
