# Offline Learning to Rank Codebase

Offline learning to rank (LTR) python codebase, algorithms below are implemented.

- *MDP-based*: DQN, DoubleDQN, **CQL**, SAC (cql_alpha = 0) 
- *baselines*: Bandit, **DLA** (state of the art)

## Project Structure

The folder structure (**bold font** for folder)

- **Click model**: simulate users' click probabilities and browsing behavior
  - **Model_files**: parameters of different user behavior, e.g. PBM, Cascade, UBM. store in `.json` files.
  - click_models.py: simulate users' behavior based on parameters in `Model_files` folder, get click lists for each queries in exps.
- **dataset**: generate initial rank lists for all queries, either using *SVM ranker* or *Random ranker* as initial ranker (logging policy)
- **libsvm_tools**: data preprocess tools, forming "ULTRA" or "ULTRE" data (not that important)
- **network**: Neural network structure for each algorithm
  - each file is named as `alg_name.py`.
- **ranker**: rankers trained with different algorithms
  - each file is named as `{alg_name}Ranker.py`.
  - general methods are: *update_policy (update policy after training one epoch)*, *validation_forward (validate on valid set while training)*
- **runs**: each file runs an exp with a certain algorithm.
  - each file is named as `run_{alg_name}.py`
  - files named `run_{dataset_name}.sh` are files that run multiple algorithms at one time, not necessary.
- **utils**: utilities, including *metric calculation* (need in validation and testing), *curves plotting*
- `svm_rank_classify`, `svm_rank_learn`, `svm_rank_linux64.tar.gz` are SVM ranker files, not that important.

## Run Experiments

All running files are under `runs` folder, each represent an algorithm. Some arguments should be given as input.

- `--feature_size`: int, dimension of each query-doc feature vector.
- `--dataset_fold`: str, folder that contains the dataset.
- `--output_fold`: str, folder that store the outputs, including training curves and trained models.
- `--data_type`: str, refers to the level of relevance label. (*"mq"* refers to 3-level and *"web10k"* refers to 5-level)
- `--logging`: str, refers to logging policy (*"svm"* or *"initial"*)
- `--five_fold`: bool, whether the dataset is divided into five-fold.
- `--test_only`: bool, train or test (default: train)

For *RL algorithms* only, 

- `--state_type`: str, indicate the design of state in MDP. 
  - *avg*: average previous document features.
  - *pos*: position embedding.
  - *pos_avg*: concatenate  of position embedding and average features.
  - *pos_avg_rew*: concatenate of position embedding, average features and previous rewards.
  - *rew*: previous rewards.
  - *avg_rew*:  concatenate of average features and previous rewards.
- `--embedding`: bool, whether using states embedding (similar to NLP)
- `--embedding_type`: str, type of embedding (*"RNN"* or *"LSTM"*)

## Results Visualization

The results are collected and visualized by `tensorboard`. Run

```bash
tensorboard --logdir {output_fold}
```

Or directly run

```bash
python ./utils/plot.py
```

to plot evaluation curves. metrics: (mrr, ndcg) @ (3,5,10)
