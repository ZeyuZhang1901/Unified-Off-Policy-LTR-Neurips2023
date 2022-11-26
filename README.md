# MDPLTR

Learning to rank python codebase, including our MDP based ones. Thanks to the codebase from https://arxiv.org/abs/2201.01534

## Results

The results are collected and visualized by `tensorboard`. Under current directory, run `tensorboard --logdir results/dataest_name/alg_name`, where,

- `dataset_name`: select from `MQ2007`, `MQ2008`...
- `alg_name`: select from `DQN`, `DoubleDQN`...

In addition, the performance of trained ranker are stored after each training session in a `.txt` file: 

- path format : `results/{dataset_name}/{alg_name}/fold{i}/{model_type}_run{j}_ndcg/performance_{setname}_{train_iteration}.txt`
  - `dataset_name`: select from `MQ2007`, `MQ2008`...
  - `alg_name`: select from `DQN`, `DoubleDQN`...
  - `i`: fold number, from 1 to 5
  - `model_type`: select from `perfect`, `informational` and `navigational`
  - `j`: run time, now is 1 only
  - `setname`: select from `trainset`, `testset`
  - `train_iteration`: now is set 100000
- e.g. `results/MQ2008/DoubleDQN/fold1/perfect_run1_ndcg/perform_testset_100000.txt`

## Run

To run the code, just run `.sh` files in `./OffpolicyLTR/run_{dataset_name}.sh`. 

- If load or save model needed, set hyperparameter `LOAD` and `SAVE` True, respectly.
