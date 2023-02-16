import sys

sys.path.append("/home/zeyuzhang/LearningtoRank/codebase/myLTR/")
whole_path = "/home/zeyuzhang/LearningtoRank/codebase/myLTR/"
from torch.utils.tensorboard import SummaryWriter
from ranker.DQN_CQLRanker import DQN_CQLRanker
from utils.input_feed import Train_Input_feed, Validation_Input_feed
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
parser.add_argument("--output_fold", type=str, required=True)
parser.add_argument("--ranker_json_file", type=str, required=True)  # ranker json file
parser.add_argument("--running_json_file", type=str, required=True)  # running json file
parser.add_argument("--start_epoch", type=int, default=0)  # start epoch
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

        print(f"Get best performance from previous training results.")
        metric_tmp = {}
        with open(checkpoint_path + "performance_test.txt") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(":")
                metric_tmp[line[0]] = float(line[1].strip())

        best_perf = metric_tmp[ranker.objective_metric]
        print(f"current best performance {ranker.objective_metric}:{best_perf}")

        print(
            f"Reload model parameters after {start_checkpoint} epochs from {checkpoint_path}"
        )
        ckpt = torch.load(checkpoint_path + f"Policy(step_{start_checkpoint}).ckpt")
        ranker.policy_net.load_state_dict(ckpt)
        ckpt = torch.load(checkpoint_path + f"Target(step_{start_checkpoint}).ckpt")
        ranker.target_net.load_state_dict(ckpt)
        if ranker.embed_type != "None":
            ckpt = torch.load(checkpoint_path + f"embed(step_{start_checkpoint}).ckpt")
            ranker.embed_model.load_state_dict(ckpt)

        ranker.global_step = start_checkpoint

    ## valid initial performance
    print(f"Checkpoint at step {ranker.global_step}")
    valid_summary = validation(valid_set, valid_input_feed, ranker)
    writer.add_scalars("Validation", valid_summary, ranker.global_step)
    for key, value in valid_summary.items():
        print(key, value)
        if not ranker.objective_metric == None:
            if key == ranker.objective_metric:
                if best_perf == None or best_perf < value:
                    best_perf = value
                    print("Save model, valid %s:%.3f" % (key, best_perf))
                    torch.save(
                        ranker.policy_net.state_dict(),
                        checkpoint_path + "Policy_best.ckpt",
                    )
                    if ranker.embed_type != "None":
                        torch.save(
                            ranker.embed_model.state_dict(),
                            checkpoint_path + "embed_best.ckpt",
                        )
                    break

    ## train and validation start
    for i in range(num_iteration - start_checkpoint):
        input_feed = train_input_feed.get_train_batch(train_set, check_validation=True)
        loss_summary, norm_summary, q_summary = ranker.update_policy(input_feed)
        writer.add_scalars("Loss", loss_summary, ranker.global_step)
        writer.add_scalars("Norm", norm_summary, ranker.global_step)
        writer.add_scalars("Q Value", q_summary, ranker.global_step)

        if (i + 1) % steps_per_checkpoint == 0:
            print(f"Checkpoint at step {ranker.global_step}")
            valid_summary = validation(valid_set, valid_input_feed, ranker)
            writer.add_scalars("Validation", valid_summary, ranker.global_step)
            for key, value in valid_summary.items():
                print(key, value)
                if not ranker.objective_metric == None:
                    if key == ranker.objective_metric:
                        if best_perf == None or best_perf < value:
                            best_perf = value
                            print("Save model, valid %s:%.3f" % (key, best_perf))
                            torch.save(
                                ranker.policy_net.state_dict(),
                                checkpoint_path + "Policy_best.ckpt",
                            )
                            if ranker.embed_type != "None":
                                torch.save(
                                    ranker.embed_model.state_dict(),
                                    checkpoint_path + "embed_best.ckpt",
                                )
                            break

            sys.stdout.flush()

        ## save model checkpoint
        if (i + 1) % steps_per_save == 0:
            print(f"Checkpoint at step {ranker.global_step} for saving")
            torch.save(
                ranker.policy_net.state_dict(),
                checkpoint_path + f"Policy(step_{ranker.global_step}).ckpt",
            )
            torch.save(
                ranker.target_net.state_dict(),
                checkpoint_path + f"Target(step_{ranker.global_step}).ckpt",
            )
            if ranker.embed_type != "None":
                torch.save(
                    ranker.embed_model.state_dict(),
                    checkpoint_path + f"embed(step_{ranker.global_step}).ckpt",
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
    performance_path,  ## used to record performance on each query
    checkpoint_path,
):
    ## Load model with best performance
    print("Reading model parameters from %s" % checkpoint_path)
    ckpt = torch.load(checkpoint_path + "Policy_best.ckpt")
    ranker.policy_net.load_state_dict(ckpt)
    ranker.policy_net.eval()
    if ranker.embed_type != "None":
        ckpt = torch.load(checkpoint_path + "embed_best.ckpt")
        ranker.embed_model.load_state_dict(ckpt)
        ranker.embed_model.eval()

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
    batch_size,
    eta,
    min_prob,
    click_type,
    rel_level,
    max_visuable_size,
    num_interaction,
    start_checkpoint,
    steps_per_checkpoint,
    steps_per_save,
    f,
    train_set,
    valid_set,
    test_set,
    ranker_json_file,
    output_fold,
    test_only,
):

    click_model_path = (
        whole_path
        + f"clickModel/{click_type}/{click_type}_{min_prob}_1.0_{rel_level}_{eta}.json"
    )
    with open(click_model_path) as fin:
        model_desc = json.load(fin)
        click_model = cm.loadModelFromJson(model_desc)

    for r in range(1, 2):
        # for r in range(1, 4):
        np.random.seed(r)
        random.seed(r)
        torch.manual_seed(r)
        print(f"Round{r}\tclick type: {click_type}\teta: {eta}\tmin prob: {min_prob}")

        if test_only:
            max_visuable_size = min(test_set.rank_list_size, max_visuable_size)
            test_input_feed = Validation_Input_feed(
                max_candidate_num=test_set.rank_list_size,
                batch_size=batch_size,
            )
            ranker = DQN_CQLRanker(
                hyper_json_file=ranker_json_file,
                feature_size=feature_size,
                rank_list_size=test_set.rank_list_size,
                max_visuable_size=max_visuable_size,
                click_model=click_model,
            )
            test(
                test_set=test_set,
                test_input_feed=test_input_feed,
                ranker=ranker,
                performance_path=f"{output_fold}/fold{f}/{click_type}/minprob_{min_prob}_eta_{eta}_run{r}/performance_test.txt",
                checkpoint_path=f"{output_fold}/fold{f}/{click_type}/minprob_{min_prob}_eta_{eta}_run{r}/",
            )

        else:
            writer = SummaryWriter(
                f"{output_fold}/fold{f}/{click_type}/minprob_{min_prob}_eta_{eta}_run{r}/"
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
            ranker = DQN_CQLRanker(
                hyper_json_file=ranker_json_file,
                feature_size=feature_size,
                rank_list_size=valid_set.rank_list_size,
                max_visuable_size=max_visuable_size,
                click_model=click_model,
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
                checkpoint_path=f"{output_fold}/fold{f}/{click_type}/minprob_{min_prob}_eta_{eta}_run{r}/",
            )
            writer.close()


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    output_fold = args.output_fold
    ranker_json_file = args.ranker_json_file
    running_json_file = args.running_json_file
    start_epoch = args.start_epoch
    test_only = args.test_only

    with open(running_json_file) as running_json:
        hypers = json.load(running_json)
    feature_size = int(hypers["feature_size"])
    max_visuable_size = int(hypers["max_visuable_size"])
    batch_size = int(hypers["batch_size"])
    dataset_fold = hypers["dataset_fold"]
    logging = hypers["logging"]
    rel_level = int(hypers["rel_level"])
    five_fold = eval(hypers["five_fold"])
    epochs = int(hypers["epochs"])
    steps_per_checkpoint = int(hypers["steps_per_checkpoint"])
    steps_per_save = int(hypers["steps_per_save"])

    click_models = hypers["click_models"]  # list
    etas = hypers["etas"]  # list
    min_probs = hypers["min_probs"]  # list

    # for 5 folds
    for f in range(1, 2):
        if logging == "svm":
            path = (
                "{}/Fold{}/tmp_data/".format(dataset_fold, f)
                if five_fold
                else "{}/tmp_data/".format(dataset_fold)
            )
        elif logging == "initial":
            path = (
                "{}/Fold{}/".format(dataset_fold, f)
                if five_fold
                else dataset_fold + "/"
            )

        if test_only:  # Test
            print("---------------------Test  start!------------------------")
            test_set = data_utils.read_data(path, "test", None, 0, logging)
            train_set, valid_set = None, None
            max_candidate_num = test_set.rank_list_size
            test_set.pad(test_set.rank_list_size)
        else:  # Train
            print("---------------------Train start!------------------------")
            print(
                f"Epochs: {epochs}\tValid step: {steps_per_checkpoint}\tSave step: {steps_per_save}"
            )
            train_set = data_utils.read_data(path, "train", None, 0, logging)
            valid_set = data_utils.read_data(path, "valid", None, 0, logging)
            test_set = None
            max_candidate_num = max(train_set.rank_list_size, valid_set.rank_list_size)
            train_set.pad(max_candidate_num)
            valid_set.pad(max_candidate_num)

        processors = []
        # for all click models
        for click_model in click_models:
            for eta in etas:
                for min_prob in min_probs:
                    p = mp.Process(
                        target=job,
                        args=(
                            feature_size,
                            batch_size,
                            eta,
                            min_prob,
                            click_model,
                            rel_level,
                            max_visuable_size,
                            epochs,
                            start_epoch,
                            steps_per_checkpoint,
                            steps_per_save,
                            f,
                            train_set,
                            valid_set,
                            test_set,
                            ranker_json_file,
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
