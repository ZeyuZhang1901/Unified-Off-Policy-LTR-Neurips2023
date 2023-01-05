import sys

sys.path.append("/home/zeyuzhang/LearningtoRank/codebase/myLTR/")
whole_path = "/home/zeyuzhang/LearningtoRank/codebase/myLTR/"
from torch.utils.tensorboard import SummaryWriter
from ranker.DoubleDQNRanker import DoubleDQNRanker
from dataset import LetorDataset
from clickModel import click_models as cm

import multiprocessing as mp
import numpy as np
import random
import torch
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--feature_size", type=int, required=True)
parser.add_argument("--dataset_fold", type=str, required=True)
parser.add_argument("--output_fold", type=str, required=True)
parser.add_argument("--data_type", type=str, required=True)  # 'mq' or 'web10k'
args = parser.parse_args()

# %%
def run(
    train_set,
    test_set,
    ranker,
    num_iteration,
    writer,
    steps_per_checkpoint,
    checkpoint_path,
):

    loss = 0.0
    loss_count = 0.0
    best_perf = None

    for i in range(num_iteration):
        input_feed, info_map = ranker.get_train_batch(train_set, check_validation=True)
        step_loss, loss_summary, norm_summary = ranker.update_policy(input_feed)
        loss += step_loss
        loss_count += 1
        writer.add_scalars("Loss", loss_summary, ranker.global_step)
        writer.add_scalars("Norm", norm_summary, ranker.global_step)

        if i % steps_per_checkpoint == 0:
            print(f"Checkpoint at step {ranker.global_step}\tLoss {loss/loss_count}")
            input_feed, info_map = ranker.get_validation_batch(
                test_set, check_validation=False
            )
            _, _, valid_summary = ranker.validation(input_feed)
            writer.add_scalars("Validation", valid_summary, ranker.global_step)
            for key, value in valid_summary.items():
                print(key, value)
                if not ranker.objective_metric == None:
                    if key == ranker.objective_metric:
                        if best_perf == None or best_perf < value:
                            best_perf = value
                            print("Save model, valid %s:%.3f" % (key, best_perf))
                            torch.save(ranker.model.state_dict(), checkpoint_path)
                            break
            loss = 0.0
            loss_count = 0
            sys.stdout.flush()


# %%
def job(
    feature_size,
    model_type,
    click_type,
    data_type,
    batch_size,
    lr,
    num_interaction,
    steps_per_checkpoint,
    target_update_steps,
    f,
    train_set,
    test_set,
    output_fold,
):

    click_model_path = (
        whole_path
        + f"clickModel/model_files/{click_type}_{data_type}_{model_type}.json"
    )
    with open(click_model_path) as fin:
        model_desc = json.load(fin)
        click_model = cm.loadModelFromJson(model_desc)

    for r in range(1, 4):
        np.random.seed(r)
        random.seed(r)
        torch.manual_seed(r)
        writer = SummaryWriter(
            "{}/fold{}/{}_{}_run{}/".format(output_fold, f, click_type, model_type, r)
        )

        ranker = DoubleDQNRanker(
            feature_dim=feature_size,
            batch_size=batch_size,
            learning_rate=lr,
            click_model=click_model,
            target_update_step=target_update_steps,
        )
        ranker.rank_list_size = min(
            ranker.rank_list_size, train_set.rank_list_size, test_set.rank_list_size
        )
        run(
            train_set=train_set,
            test_set=test_set,
            ranker=ranker,
            num_iteration=num_interaction,
            writer=writer,
            steps_per_checkpoint=steps_per_checkpoint,
            checkpoint_path="{}/fold{}/{}_{}_run{}/DQN.ckpt".format(
                output_fold, f, click_type, model_type, r
            ),
        )
        writer.close()


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    END_POS = 10
    FEATURE_SIZE = args.feature_size
    BATCH_SIZE = 256
    NUM_INTERACTION = 30000
    STEPS_PER_CHECKPOINT = 100
    TARGET_UPDATE_STEPS = 50
    LR = 1e-3
    NORMALIZE = False
    DISCOUNT = 0.9
    DATA_TYPE = args.data_type

    # model_types = ["informational", "perfect", "navigational"]
    model_types = ["informational", "perfect"]
    # model_types = ["perfect"]
    click_types = ["pbm", "cascade"]
    # click_types = ["pbm"]

    dataset_fold = args.dataset_fold
    output_fold = args.output_fold

    # for 5 folds
    for f in range(1, 2):
        training_path = "{}/Fold{}/train.txt".format(dataset_fold, f)
        test_path = "{}/Fold{}/test.txt".format(dataset_fold, f)
        train_set = LetorDataset(
            training_path, FEATURE_SIZE, query_level_norm=NORMALIZE
        )
        test_set = LetorDataset(test_path, FEATURE_SIZE, query_level_norm=NORMALIZE)

        processors = []
        # for 3 click_models
        for click_type in click_types:
            for mode_type in model_types:
                p = mp.Process(
                    target=job,
                    args=(
                        FEATURE_SIZE,
                        mode_type,
                        click_type,
                        DATA_TYPE,
                        BATCH_SIZE,
                        LR,
                        NUM_INTERACTION,
                        STEPS_PER_CHECKPOINT,
                        TARGET_UPDATE_STEPS,
                        f,
                        train_set,
                        test_set,
                        output_fold,
                    ),
                )
                p.start()
                processors.append(p)
    for p in processors:
        p.join()
