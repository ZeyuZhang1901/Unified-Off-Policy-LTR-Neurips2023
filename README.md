# Offline Learning to Rank Codebase

Offline learning to rank (LTR) python codebase, algorithms below are or will be implemented.

- *MDP-based*: DQN, DoubleDQN, BCQ, CQL, SAC 
- *baselines*: Bandit, DLA (state of the art)

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

## Results Visualization

The results are collected and visualized by `tensorboard`. Run `tensorboard --logdir {output_fold}` directly.
