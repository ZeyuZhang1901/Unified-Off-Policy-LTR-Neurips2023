import sys

sys.path.append("/home/zeyuzhang/LearningtoRank/codebase/myLTR/")
whole_path = "/home/zeyuzhang/LearningtoRank/codebase/myLTR/"
from torch.utils.tensorboard import SummaryWriter
from ranker.DLARanker import DLARanker
from ranker.input_feed import Train_Input_feed, Validation_Input_feed
from dataset import data_utils

from clickModel import click_models as cm

import torch.multiprocessing as mp
import numpy as np
import random
import torch
import json
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--feature_size", type=int, required=True)
parser.add_argument("--dataset_fold", type=str, required=True)
parser.add_argument("--output_fold", type=str, required=True)
parser.add_argument("--data_type", type=str, required=True)  # 'mq' or 'web10k'
parser.add_argument("--logging", type=str, required=True)  ## 'svm' or 'initial'
parser.add_argument("--five_fold", default=True, action="store_true")  # fivefold
parser.add_argument("--test_only", default=False, action="store_true")  # train or test
args = parser.parse_args()


def merge_Summary(summary_list, weights):
    merged_values = {}
    weight_sum_map = {}
    for i in range(len(summary_list)):
        summary = summary_list[i]
        for metric in summary.keys():
            if metric not in merged_values:
                merged_values[metric] = 0.0
                weight_sum_map[metric] = 0.0
            merged_values[metric] += summary[metric] * weights[i]
            weight_sum_map[metric] += weights[i]
    for k in merged_values:
        merged_values[k] /= max(0.0000001, weight_sum_map[k])
    return merged_values


# %%
def train(
    train_set,
    valid_set,
    train_input_feed,
    valid_input_feed,
    ranker,
    num_iteration,
    start_checkpoint,  # (int) if 0, means train from scratch. Must be an integer multiple of `steps_per_save`
    writer,
    steps_per_checkpoint,
    steps_per_save,
    checkpoint_path,
):

    best_perf = None

    ## Load actor, critic1 and critic2 statistics from selected checkpoints
    if start_checkpoint > 0:
        assert (
            start_checkpoint % steps_per_save == 0
        ), "Invalid start checkpoint! Must be an integer multiple of `steps_per_save`"

        print(
            f"Reload model parameters after {start_checkpoint} epochs from {checkpoint_path}",flush=True
        )
        ckpt = torch.load(checkpoint_path + f"DLA(step_{start_checkpoint}).ckpt")
        ranker.model.load_state_dict(ckpt)
        ckpt = torch.load(
            checkpoint_path + f"DLA_propensity(step_{start_checkpoint}).ckpt"
        )
        ranker.propensity_model.load_state_dict(ckpt)

        ranker.global_step = start_checkpoint

    ## valid initial performance
    print(f"Checkpoint at step {ranker.global_step}",flush=True)
    valid_summary = validation(valid_set, valid_input_feed, ranker)
    writer.add_scalars("Validation", valid_summary, ranker.global_step)
    for key, value in valid_summary.items():
        print(key, value,flush=True)
        if not ranker.objective_metric == None:
            if key == ranker.objective_metric:
                if best_perf == None or best_perf < value:
                    best_perf = value
                    print("Save model, valid %s:%.3f" % (key, best_perf),flush=True)
                    torch.save(
                        ranker.model.state_dict(),
                        checkpoint_path + "DLA_best.ckpt",
                    )
                    break

    ## train and validation start
    for i in range(num_iteration - start_checkpoint):
        input_feed = train_input_feed.get_train_batch(train_set, check_validation=True)
        loss_summary, norm_summary = ranker.update_policy(input_feed)
        writer.add_scalars("Loss", loss_summary, ranker.global_step)
        writer.add_scalars("Norm", norm_summary, ranker.global_step)

        if (i + 1) % steps_per_checkpoint == 0:
            print(f"Checkpoint at step {ranker.global_step}",flush=True)
            valid_summary = validation(valid_set, valid_input_feed, ranker)
            writer.add_scalars("Validation", valid_summary, ranker.global_step)
            for key, value in valid_summary.items():
                print(key, value,flush=True)
                if not ranker.objective_metric == None:
                    if key == ranker.objective_metric:
                        if best_perf == None or best_perf < value:
                            best_perf = value
                            print("Save model, valid %s:%.3f" % (key, best_perf),flush=True)
                            torch.save(
                                ranker.model.state_dict(),
                                checkpoint_path + "DLA_best.ckpt",
                            )
                            break

            sys.stdout.flush()

        ## save model checkpoint
        if (i + 1) % steps_per_save == 0:
            print(f"Checkpoint at step {ranker.global_step} for saving",flush=True)
            torch.save(
                ranker.model.state_dict(),
                checkpoint_path + f"DLA(step_{ranker.global_step}).ckpt",
            )
            torch.save(
                ranker.propensity_model.state_dict(),
                checkpoint_path + f"DLA_propensity(step_{ranker.global_step}).ckpt",
            )


def validation(
    dataset,
    data_input_feed,
    ranker,
):
    offset = 0
    count_batch = 0.0
    summary_list = []
    batch_size_list = []
    while offset < len(dataset.initial_list):
        input_feed = data_input_feed.get_validation_batch(
            offset, dataset, check_validation=False
        )
        _, _, summary = ranker.validation(input_feed)

        ## deepcopy the summary dict
        summary_list.append(copy.deepcopy(summary))
        batch_size_list.append(len(input_feed["docid_input0"]))
        offset += batch_size_list[-1]
        count_batch += 1.0
    valid_summary = merge_Summary(summary_list, batch_size_list)
    return valid_summary


def test(
    test_set,
    test_input_feed,
    ranker,
    performance_path,  # used to record performance on each query
    checkpoint_path,
):
    ## Load model with best performance
    print("Reading model parameters from %s" % checkpoint_path, flush=True)
    ckpt = torch.load(checkpoint_path + "DLA_best.ckpt")
    ranker.model.load_state_dict(ckpt)
    ranker.model.eval()

    with torch.no_grad():
        test_summary = validation(test_set, test_input_feed, ranker)

    ## record in `performance_path`.txt file
    lines = []
    for metric, value in test_summary.items():
        lines.append(f"{metric}: {value}\n")
    with open(performance_path, "w") as fin:
        fin.writelines(lines)


# %%
def job(
    feature_size,
    model_type,
    click_type,
    data_type,
    batch_size,
    lr,
    max_visuable_size,
    metric_type,
    metric_topn,
    num_interaction,
    start_checkpoint,
    steps_per_checkpoint,
    steps_per_save,
    f,
    train_set,
    valid_set,
    test_set,
    output_fold,
    test_only,
):
    click_model_path = (
        whole_path
        + f"clickModel/model_files/{click_type}_{data_type}_{model_type}.json"
    )
    with open(click_model_path) as fin:
        model_desc = json.load(fin)
        click_model = cm.loadModelFromJson(model_desc)

    # for r in range(1, 4):
    for r in range(1, 2):
        np.random.seed(r)
        random.seed(r)
        torch.manual_seed(r)
        print(f"Round{r}\tclick type: {click_type}\tmodel type: {model_type}",flush=True)

        if not test_only:
            writer = SummaryWriter(
                "{}/fold{}/{}_{}_run{}/".format(
                    output_fold, f, click_type, model_type, r
                )
            )
            max_visuable_size = min(
                train_set.rank_list_size, valid_set.rank_list_size, max_visuable_size
            )
            train_input_feed = Train_Input_feed(
                click_model=click_model,
                max_visuable_size=max_visuable_size,
                batch_size=batch_size,
            )
            valid_input_feed = Validation_Input_feed(
                max_candidate_num=valid_set.rank_list_size,
                batch_size=batch_size,
            )
            ranker = DLARanker(
                feature_size=feature_size,
                click_model=click_model,
                learning_rate=lr,
                batch_size=batch_size,
                rank_list_size=valid_set.rank_list_size,
                metric_type=metric_type,
                metric_topn=metric_topn,
            )
            train(
                train_set=train_set,
                valid_set=valid_set,
                train_input_feed=train_input_feed,
                valid_input_feed=valid_input_feed,
                ranker=ranker,
                num_iteration=num_interaction,
                start_checkpoint=start_checkpoint,
                writer=writer,
                steps_per_checkpoint=steps_per_checkpoint,
                steps_per_save=steps_per_save,
                checkpoint_path="{}/fold{}/{}_{}_run{}/".format(
                    output_fold, f, click_type, model_type, r
                ),
            )
            writer.close()
        
        max_visuable_size = min(test_set.rank_list_size, max_visuable_size)
        test_input_feed = Validation_Input_feed(
            max_candidate_num=test_set.rank_list_size,
            batch_size=batch_size,
        )
        ranker = DLARanker(
            feature_size=feature_size,
            click_model=click_model,
            learning_rate=lr,
            batch_size=batch_size,
            rank_list_size=test_set.rank_list_size,
            metric_type=metric_type,
            metric_topn=metric_topn,
        )
        test(
            test_set=test_set,
            test_input_feed=test_input_feed,
            ranker=ranker,
            performance_path="{}/fold{}/{}_{}_run{}/performance_test.txt".format(
                output_fold, f, click_type, model_type, r
            ),
            checkpoint_path="{}/fold{}/{}_{}_run{}/".format(
                output_fold, f, click_type, model_type, r
            ),
        )



# %%
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    MAX_VISUABLE_POS = 10
    FEATURE_SIZE = args.feature_size
    BATCH_SIZE = 256
    NUM_INTERACTION = 10000
    # NUM_INTERACTION = 30000
    STEPS_PER_SAVE = 1000
    STEPS_PER_CHECKPOINT = 50
    START_CHECKPOINT = 0  # usually start from scratch
    LR = 1e-5

    DATA_TYPE = args.data_type
    LOGGING = args.logging  ## logging policy type

    metric_type = ["mrr", "ndcg"]
    metric_topn = [3, 5, 10]
    objective_metric = "ndcg_10"

    # model_types = ["informational", "perfect", "navigational"]
    # model_types = ["informational", "perfect"]
    model_types = ["perfect"]
    # click_types = ["pbm", "cascade"]
    # click_types = ["pbm"]
    click_types = ["cascade"]

    dataset_fold = args.dataset_fold
    output_fold = args.output_fold
    five_fold = args.five_fold
    test_only = args.test_only  # whether train or test

    # for 5 folds
    for f in range(1, 2):
        if LOGGING == "svm":
            path = (
                "{}/Fold{}/tmp_data/".format(dataset_fold, f)
                if five_fold
                else "{}/tmp_data/".format(dataset_fold)
            )
        elif LOGGING == "initial":
            path = (
                "{}/Fold{}/".format(dataset_fold, f)
                if five_fold
                else dataset_fold + "/"
            )

        if not test_only:  # Train
            print("-------------------- Training------------------------", flush=True)
            print(
                f"Epochs: {NUM_INTERACTION}\tValid step: {STEPS_PER_CHECKPOINT}\tSave step: {STEPS_PER_SAVE}", flush=True
            )
            train_set = data_utils.read_data(path, "train", None, 0, LOGGING)
            valid_set = data_utils.read_data(path, "valid", None, 0, LOGGING)
            max_candidate_num = max(train_set.rank_list_size, valid_set.rank_list_size)
            train_set.pad(max_candidate_num)
            valid_set.pad(max_candidate_num)
        
        test_set = data_utils.read_data(path, "test", None, 0, LOGGING)
        test_set.pad(test_set.rank_list_size)

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
                        MAX_VISUABLE_POS,
                        metric_type,
                        metric_topn,
                        NUM_INTERACTION,
                        START_CHECKPOINT,
                        STEPS_PER_CHECKPOINT,
                        STEPS_PER_SAVE,
                        f,
                        train_set,
                        valid_set,
                        test_set,
                        output_fold,
                        test_only,
                    ),
                )
                p.start()
                processors.append(p)
        if not five_fold:  # if not using five-fold validation
            break
    for p in processors:
        p.join()
