import sys
sys.path.append('/home/zeyuzhang/LearningtoRank/codebase/myLTR/')
sys.path.append('../')
from torch.utils.tensorboard import SummaryWriter
from utils import evl_tool
from network.Memory import Memory
from ranker.BCQRanker import BCQRanker
from ranker.RandomRanker import RandomRanker
from clickModel.CM import CM
from clickModel.PBM import PBM
from data_collect import dataCollect
from dataset import LetorDataset

import multiprocessing as mp
import numpy as np
np.random.seed(1958)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--feature_size', type=int, required=True)
parser.add_argument('--dataset_fold', type=str, required=True)
parser.add_argument('--output_fold', type=str, required=True)
parser.add_argument('--rel_level', type=int, required=True)
parser.add_argument('--sample_iter', type=int, required=True)
args = parser.parse_args()

# %%


def run(test_set,
        ranker,
        memory,
        num_iteration,
        end_pos
        ):

    ndcg_scores = []
    q1_values = []
    q2_values = []
    target_q1_values = []
    target_q2_values = []
    actor_losses = []
    critic_losses = []

    for i in range(num_iteration):
        q1, q2, target_q1, target_q2, critic_loss, actor_loss = ranker.update_policy(
            memory)
        q1_values.append(q1)
        q2_values.append(q2)
        target_q1_values.append(target_q1)
        target_q2_values.append(target_q2)
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)

        print(f"iter {i+1}: ")
        print(f"q1_value {q1} q2_value {q2}")
        print(f"target_q1_value {target_q1} target_q2_value {target_q2}")
        print(f"actor_loss {actor_loss} critic_loss {critic_loss}")
        # evaluate 100 times in one train session
        if i % int(num_iteration/100) == 0:
            all_result = ranker.get_all_query_result_list(test_set)
            ndcg = evl_tool.average_ndcg_at_k(test_set, all_result, end_pos)
            ndcg_scores.append(ndcg)
            print(f"eval {int((i+1)/100)} ndcg {ndcg}")

    return ndcg_scores, q1_values, q2_values, target_q1_values, \
        target_q2_values, actor_losses, critic_losses

# %%


def job(model_type,
        sample_iteration,
        state_dim,
        action_dim,
        memory,
        f,
        train_set,
        test_set,
        output_fold,
        load=False,
        save=False
        ):

    if args.rel_level == 5:
        if model_type == "perfect":
            pc = [0.0, 0.2, 0.4, 0.8, 1.0]
            ps = [0.0, 0.0, 0.0, 0.0, 0.0]
        elif model_type == "navigational":
            pc = [0.05, 0.3, 0.5, 0.7, 0.95]
            ps = [0.2, 0.3, 0.5, 0.7, 0.9]
        elif model_type == "informational":
            pc = [0.4, 0.6, 0.7, 0.8, 0.9]
            ps = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif args.rel_level == 3:
        if model_type == "perfect":
            pc = [0.0, 0.5, 1.0]
            ps = [0.0, 0.0, 0.0]
        elif model_type == "navigational":
            pc = [0.05, 0.5, 0.95]
            ps = [0.2, 0.5, 0.9]
        elif model_type == "informational":
            pc = [0.4, 0.7, 0.9]
            ps = [0.1, 0.3, 0.5]

    # cm = PBM(pc, 1)
    cm = CM(pc, 1)

    for r in range(1, 2):
        # np.random.seed(r)
        writer = SummaryWriter(
            "{}/fold{}/{}_run{}_ndcg/".format(output_fold, f, model_type, r))
        print("DQN fold{} {}  run{} start!".format(f, model_type, r))

        ranker = BCQRanker(state_dim, action_dim, MAX_ACTION, BATCH_SIZE, LR,
                           DISCOUNT, TAU, LAMBDA, PHI)
        behavior_ranker = RandomRanker()
        if load:
            ranker.load_ranker(
                f'{output_fold}/fold{f}/{model_type}_run{r}_ndcg/')
        dataCollect(state_dim, action_dim, memory, behavior_ranker,
                    train_set, cm, sample_iteration, CAPACITY, END_POS)
        ndcg_scores, q1_values, q2_values, target_q1_values, \
            target_q2_values, actor_losses, critic_losses = run(
                test_set, ranker, memory, NUM_INTERACTION, END_POS)
        if save:
            ranker.restore_ranker(
                f'{output_fold}/fold{f}/{model_type}_run{r}_ndcg/')

        evl_tool.write_performance(path=f'{output_fold}/fold{f}/{model_type}_run{r}_ndcg/perform_trainset_{NUM_INTERACTION}.txt',
                                   dataset=train_set, ranker=ranker, end_pos=END_POS)
        evl_tool.write_performance(path=f'{output_fold}/fold{f}/{model_type}_run{r}_ndcg/perform_testset_{NUM_INTERACTION}.txt',
                                   dataset=test_set, ranker=ranker, end_pos=END_POS)
        print(f"matrics record start!")
        for i in range(len(ndcg_scores)):
            writer.add_scalar('ndcg', ndcg_scores[i], i+1)
        for j in range(len(q1_values)):
            writer.add_scalars('policy', {'q1': q1_values[j],
                                          'q2': q2_values[j]}, j+1)
            writer.add_scalars('target', {'target_q1': target_q1_values[j],
                                         'target_q2': target_q2_values[j]}, j+1)
            writer.add_scalars('avg_loss', {'actor_loss': actor_losses[i],
                                            'critic_loss': critic_losses[j]}, j+1)
        print(f"matrics record finish!")
        writer.close()


if __name__ == "__main__":

    END_POS = 10
    FEATURE_SIZE = args.feature_size
    STATE_DIM = FEATURE_SIZE + END_POS * 2  # record previous rewards (cascade) and position (position)
    MAX_ACTION = 1.0  # after normalization
    BATCH_SIZE = 256
    NUM_INTERACTION = 100000
    SAMPLE_ITERATION = args.sample_iter
    CAPACITY = 3e6
    DISCOUNT = 0.9
    TAU = 0.005
    LAMBDA = 0.75  # target calculation
    PHI = 0.05  # perturb range
    LR = 1e-3
    LOAD = False
    SAVE = True
    NORMALIZE = True

    click_models = ["informational", "perfect", "navigational"]
    # click_models = ["informational", "perfect"]
    # click_models = ["perfect"]

    dataset_fold = args.dataset_fold
    output_fold = args.output_fold

    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(
            training_path, FEATURE_SIZE, query_level_norm=NORMALIZE)
        test_set = LetorDataset(test_path, FEATURE_SIZE,
                                query_level_norm=NORMALIZE)
        memory = Memory(capacity=int(CAPACITY))

        processors = []
        # for 3 click_models
        for click_model in click_models:
            p = mp.Process(target=job,
                           args=(click_model, SAMPLE_ITERATION, STATE_DIM, FEATURE_SIZE, memory,
                                 f, train_set, test_set, output_fold, LOAD, SAVE))
            p.start()
            processors.append(p)
    for p in processors:
        p.join()
