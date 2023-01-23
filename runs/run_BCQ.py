import sys

sys.path.append("/home/zeyuzhang/LearningtoRank/codebase/myLTR/")
whole_path = "/home/zeyuzhang/LearningtoRank/codebase/myLTR/"
from torch.utils.tensorboard import SummaryWriter
from ranker.BCQRanker import BCQRanker
from ranker import input_feed
from dataset import LetorDataset
from clickModel import click_models as cm
from utils import metrics

import torch.multiprocessing as mp
import numpy as np
import random
import torch
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--feature_size", type=int, required=True)
parser.add_argument("--dataset_fold", type=str, required=True)
parser.add_argument("--output_fold", type=str, required=True)
parser.add_argument("--data_type", type=str, required=True)  # 'mq' or 'web10k'
args = parser.parse_args()

# %%
def train(
    train_set,
    valid_set,
    train_input_feed,
    valid_input_feed,
    ranker,
    num_iteration,
    writer,
    steps_per_checkpoint,
    steps_per_save,
    checkpoint_path,
):

    loss = 0.0
    loss_count = 0.0
    best_perf = None

    for i in range(num_iteration):
        input_feed = train_input_feed.get_train_batch(train_set, check_validation=True)
        step_loss, loss_summary, norm_summary, q_summary = ranker.update_policy(
            input_feed
        )
        loss += step_loss
        loss_count += 1
        writer.add_scalars("Loss", loss_summary, ranker.global_step)
        writer.add_scalars("Norm", norm_summary, ranker.global_step)
        writer.add_scalars("Q Value", q_summary, ranker.global_step)

        if i % steps_per_checkpoint == 0:
            print(f"Checkpoint at step {ranker.global_step}")
            input_feed = valid_input_feed.get_validation_batch(
                valid_set, check_validation=False
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
                            torch.save(
                                ranker.model.state_dict(),
                                checkpoint_path + "BCQ_best.ckpt",
                            )
                            break
            loss = 0.0
            loss_count = 0
            sys.stdout.flush()

        if i % steps_per_save == 0:
            print(f"Checkpoint at step {ranker.global_step} for saving")
            torch.save(
                ranker.model.state_dict(),
                checkpoint_path + f"BCQ(step_{ranker.global_step}).ckpt",
            )


def test(
    test_set,
    test_input_feed,
    ranker,
    metric_type,
    metric_topn,
    performance_path,  # used to record performance on each query
    checkpoint_path,
):
    ## Load model with best performance
    print("Reading model parameters from %s" % checkpoint_path + "BCQ_best.ckpt")
    ckpt = torch.load(checkpoint_path + "BCQ_best.ckpt")
    ranker.model.load_state_dict(ckpt)
    ranker.model.eval()

    ## test performance and write labels
    input_feed_test = test_input_feed.get_validation_batch(
        test_set, check_validation=False
    )
    (
        ranker.docid_inputs,
        ranker.letor_features,
        ranker.labels,
    ) = input_feed.create_input_feed(
        input_feed_test, ranker.rank_list_size, ranker.device
    )

    metric_record = {}
    with torch.no_grad():
        ranker.output, labels = ranker.validation_forward()
        pad_removed_output = ranker.remove_padding_for_metric_eval()
        for metric in metric_type:
            topns = metric_topn
            metric_values = metrics.make_ranking_metric_fn(metric, topns)(
                ranker.labels, pad_removed_output, None
            )
            for topn, metric_value in zip(topns, metric_values):
                metric_record[f"{metric}_{topn}"] = metric_value.item()

    ## record in `performance_path`.txt file
    qids = test_set.get_all_querys()
    lines = []
    labels_list = torch.unbind(labels.to(torch.int64), dim=0)
    sorted_labels = labels.sort(descending=True, dim=1).values
    sorted_labels_list = torch.unbind(sorted_labels.to(torch.int64), dim=0)
    for metric, value in metric_record.items():
        lines.append(f"{metric}: {value}\n")
    lines.append("\nRank lists and ideal rank lists for each query:\n")
    for i in range(len(qids)):
        lines.append(f"qid: {qids[i]}\n")
        lines.append(f"policy_label_list\t{labels_list[i].tolist()}\n")
        lines.append(f"sorted_label_list\t{sorted_labels_list[i].tolist()}\n")

    with open(performance_path, "a") as fin:
        fin.writelines(lines)


# %%
def job(
    feature_size,
    model_type,
    click_type,
    data_type,
    batch_size,
    lr,
    bcq_threshold,
    max_visuable_size,
    metric_type,
    metric_topn,
    num_interaction,
    steps_per_checkpoint,
    steps_per_save,
    target_update_steps,
    f,
    train_set,
    valid_set,
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
        print(f"{r} Train start! click type: {click_type}\tmodel type: {model_type}")
        np.random.seed(r)
        random.seed(r)
        torch.manual_seed(r)
        writer = SummaryWriter(
            "{}/fold{}/{}_{}_run{}/".format(output_fold, f, click_type, model_type, r)
        )

        max_visuable_size = min(
            train_set.rank_list_size, valid_set.rank_list_size, max_visuable_size
        )
        train_input_feed = input_feed.Train_Input_feed(
            click_model=click_model,
            max_visuable_size=max_visuable_size,
            batch_size=batch_size,
        )
        valid_input_feed = input_feed.Validation_Input_feed(
            max_candidate_num=valid_set.rank_list_size
        )
        ranker = BCQRanker(
            feature_dim=feature_size,
            batch_size=batch_size,
            learning_rate=lr,
            max_visuable_size=max_visuable_size,
            rank_list_size=valid_set.rank_list_size,
            metric_type=metric_type,
            metric_topn=metric_topn,
            click_model=click_model,
            target_update_step=target_update_steps,
            bcq_threshold=bcq_threshold,
        )
        train(
            train_set=train_set,
            valid_set=valid_set,
            train_input_feed=train_input_feed,
            valid_input_feed=valid_input_feed,
            ranker=ranker,
            num_iteration=num_interaction,
            writer=writer,
            steps_per_checkpoint=steps_per_checkpoint,
            steps_per_save=steps_per_save,
            checkpoint_path="{}/fold{}/{}_{}_run{}/".format(
                output_fold, f, click_type, model_type, r
            ),
        )
        writer.close()

        print(f"{r} Test start! click type: {click_type}\tmodel type: {model_type}")
        test_input_feed = input_feed.Validation_Input_feed(
            max_candidate_num=test_set.rank_list_size
        )
        test(
            test_set=test_set,
            test_input_feed=test_input_feed,
            ranker=ranker,
            metric_type=metric_type,
            metric_topn=metric_topn,
            performance_path="{}/fold{}/{}_{}_run{}/performance_test.txt".format(
                output_fold, f, click_type, model_type, r
            ),
            checkpoint_path="{}/fold{}/{}_{}_run{}/".format(
                output_fold, f, click_type, model_type, r
            ),
        )


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    MAX_VISUABLE_POS = 10
    FEATURE_SIZE = args.feature_size
    BATCH_SIZE = 256
    NUM_INTERACTION = 10000
    # NUM_INTERACTION = 30000
    STEPS_PER_CHECKPOINT = 100
    STEPS_PER_SAVE = 1000
    TARGET_UPDATE_STEPS = 50
    LR = 1e-3
    BCQ_THRESHOLD = 0.3
    NORMALIZE = False
    DISCOUNT = 0.9
    DATA_TYPE = args.data_type

    # model_types = ["informational", "perfect", "navigational"]
    model_types = ["informational", "perfect"]
    # model_types = ["perfect"]
    click_types = ["pbm", "cascade"]
    # click_types = ["pbm"]
    # click_types = ["cascade"]

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
                        BCQ_THRESHOLD,
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
