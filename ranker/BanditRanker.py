import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from ranker.AbstractRanker import AbstractRanker
from ranker.input_feed import create_input_feed
from network.Bandit import Bandit
from utils import metrics


class BanditRanker(AbstractRanker):
    def __init__(
        self,
        feature_dim,
        click_model,  # click model used to generate clicks when constructing batch
        learning_rate,
        rank_list_size,  # considered max length of each rank list,
        metric_type,
        metric_topn,
        objective_metric="ndcg_10",
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        batch_size=256,
        max_visuable_size=10,  # max length user can see
        # dynamic_bias_eta_change=0.0,  # Set eta change step for dynamic bias severity in training, 0.0 means no change
        # dynamic_bias_step_interval=1000,  # Set how many steps to change eta for dynamic bias severity in training, 0.0 means no change
        # l2_loss=1e-5,  # Set strength for L2 regularization.
        l2_loss=0.0,  # Set strength for L2 regularization.
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_dim = feature_dim
        self.l2_loss = l2_loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        # self.dynamic_bias_eta_change = dynamic_bias_eta_change
        # self.dynamic_bias_step_interval = dynamic_bias_step_interval
        self.max_gradient_norm = max_gradient_norm
        self.click_model = click_model
        self.model = Bandit(self.feature_dim).to(self.device)
        self.optimizer_func = torch.optim.Adam
        self.loss_func = F.mse_loss

        self.metric_type = metric_type
        self.metric_topn = metric_topn
        self.objective_metric = objective_metric

        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.score_summary = {}
        self.eval_summary = {}

    # %%
    # TODO Run Model
    def get_scores(
        self,
        input_id_list,
    ):

        # Build feature padding
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[i], 0)).to(
                    self.device
                )
            )
        return self.model.forward_current(input_feature_list)

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.model.train()
        scores = self.get_scores(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )  # tensor with shape [batch_size x max_visuable_size, 1]

        labels = torch.transpose(self.labels, 0,1).reshape(-1,1)
        self.loss = self.loss_func(scores, labels)

        # update
        self.separate_gradient_update()

        # summary
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        self.score_summary["Avg Score"] = scores.mean()
        print(
            f"Step {self.global_step}\tLoss {self.loss}\tGradient Norm {self.norm}\tAvg Score {scores.mean()}"
        )
        self.global_step += 1

        return self.loss_summary, self.norm_summary, self.score_summary

    def separate_gradient_update(self):
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.loss += self.l2_loss * torch.sum(p**2) / 2

        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)
        opt_ranker.zero_grad()
        self.loss.backward()

        if self.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

        opt_ranker.step()

        total_norm = 0

        for p in ranking_model_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.norm = total_norm

    # %%
    # TODO Validation step and its auxiliary functions
    def validation(self, input_feed):
        self.model.eval()
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.rank_list_size, self.device
        )
        with torch.no_grad():
            self.output = self.validation_forward()
        pad_removed_output = self.remove_padding_for_metric_eval()
        ## reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        for metric in self.metric_type:
            topns = self.metric_topn
            metric_values = metrics.make_ranking_metric_fn(metric, topns)(
                self.labels, pad_removed_output, None
            )
            for topn, metric_value in zip(topns, metric_values):
                self.eval_summary[f"{metric}_{topn}"] = metric_value.item()

        return None, self.output, self.eval_summary  # no loss, outputs, summary.

    def validation_forward(self):
        local_batch_size = self.docid_inputs.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        candidates_list = []

        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(
                    np.take(self.letor_features, self.docid_inputs[:, i], 0)
                ).to(self.device)
            )
        return self.model.forward(candidates_list)

    def remove_padding_for_metric_eval(self):
        output_scores = torch.unbind(self.output, dim=1)
        if len(output_scores) > len(self.docid_inputs):
            raise AssertionError(
                "Input id list is shorter than output score list when remove padding."
            )
        # Build mask
        valid_flags = torch.cat(
            (torch.ones(self.letor_features.shape[0] - 1), torch.zeros([1])), dim=0
        )
        valid_flags = valid_flags.type(torch.bool)
        input_flag_list = []
        for i in range(len(output_scores)):
            index_to_remove = torch.index_select(
                input=valid_flags, dim=0, index=self.docid_inputs[i]
            )
            input_flag_list.append(index_to_remove)
        # Mask padding documents
        output_scores = list(output_scores)
        for i in range(len(output_scores)):
            output_scores[i] = torch.where(
                input_flag_list[i].to(self.device),
                output_scores[i],
                torch.ones_like(output_scores[i], device=self.device) * (-100000),
            )
        return torch.stack(output_scores, dim=1)
