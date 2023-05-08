import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json

from ranker.AbstractRanker import AbstractRanker
from utils.input_feed import create_input_feed
from network.LambdaRank import DNN
from utils import metrics


class LambdaRanker(AbstractRanker):
    def __init__(
        self,
        hyper_json_file,
        feature_size,
        rank_list_size,
        max_visuable_size,
        click_model,
    ):
        ## load hyperparameters from json files
        with open(hyper_json_file) as ranker_json:
            hypers = json.load(ranker_json)
        self.batch_size = hypers["batch_size"]
        self.policy_lr = hypers["policy_lr"]
        self.EM_step_size = hypers["EM_step_size"]  # Step size for EM algorithm
        self.regulation_p = hypers["regulation_p"]  # Set strength for L2 regularization
        self.sigma = hypers["sigma"]  # Set sigma for the Gaussian kernel
        self.max_gradient_norm = hypers["max_gradient_norm"]  # Set gradient clip
        self.metric_type = hypers["metric_type"]  # Set metric type for evaluation
        self.metric_topn = hypers["metric_topn"]  # Set topn for evaluation
        self.objective_metric = hypers[
            "objective_metric"
        ]  # Set objective metric for training

        ## parameters from function interface
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        self.click_model = click_model

        self.model = DNN(feature_size=self.feature_size).to(self.device)
        self.optimizer_func = torch.optim.Adam

        self.t_plus = torch.ones(
            [1, self.rank_list_size], dtype=torch.float32, device=self.device
        )
        self.t_minus = torch.ones(
            [1, self.rank_list_size], dtype=torch.float32, device=self.device
        )
        self.t_plus.requires_grad = False
        self.t_minus.requires_grad = False

        self.global_batch_count = 0
        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.eval_summary = {}

    def ranking_model(self):
        output_scores = self.get_ranking_scores(input_id_list=self.docid_inputs)
        return torch.cat(output_scores, 1)

    def get_ranking_scores(
        self,
        input_id_list,
    ):
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(letor_features, input_id_list[i], 0))
            )
        return self.model.forward(input_feature_list)

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.model.train()
        self.splitted_t_plus = torch.split(self.t_plus, 1, dim=1)
        training_output = self.ranking_model()
        preds_sorted, preds_sorted_inds = torch.sort(
            training_output, dim=1, descending=True
        )
        labels_sorted_via_preds = torch.gather(self.labels, 1, preds_sorted_inds)

        std_diffs = torch.unsqueeze(labels_sorted_via_preds, dim=2) - torch.unsqueeze(
            labels_sorted_via_preds, dim=1
        )  # standard pairwise differences S_ij
        std_Sij = torch.clamp(std_diffs, min=-1, max=1)  # clip S_ij into [-1, 1]
        std_p_ij = 0.5 * (1 + std_Sij)  # standard pairwise probabilities P_ij
        # till now s_ij has shape [batch_size, rank_list_size, rank_list_size]
        s_ij = torch.unsqueeze(preds_sorted, dim=2) - torch.unsqueeze(
            preds_sorted, dim=1
        )  # computing pairwise differences, s_i - s_j
        p_ij = 1.0 / (torch.exp(self.sigma * s_ij) + 1.0)  # pairwise probabilities
        ideally_sorted_labels, _ = torch.sort(self.labels, dim=1, descending=True)
        delta_NDCG = self.compute_delta_ndcg(
            ideally_sorted_labels, labels_sorted_via_preds
        )

        self.loss = nn.BCEWithLogitsLoss(delta_NDCG, reduction="none")(p_ij, std_p_ij)
        pair_loss = torch.sum(self.loss, 0)
        t_plus_loss_list = torch.sum(pair_loss / self.t_minus, 1)
        pair_loss_ji = torch.transpose(pair_loss, 0, 1)
        t_minus_loss_list = torch.sum(pair_loss_ji / self.t_plus, 1)
        t_plus_t_minus = torch.unsqueeze(self.t_plus, dim=2) * self.t_minus
        pair_loss_debias = metrics._safe_div(pair_loss, t_plus_t_minus)
        self.loss = torch.sum(pair_loss_debias)

        with torch.no_grad():
            self.t_plus = (
                1 - self.EM_step_size
            ) * self.t_plus + self.EM_step_size * torch.pow(
                metrics._safe_div(t_plus_loss_list, t_plus_loss_list[0]),
                1 / (self.regulation_p + 1),
            )
            self.t_minus = (
                1 - self.EM_step_size
            ) * self.t_minus + self.EM_step_size * torch.pow(
                metrics._safe_div(t_minus_loss_list, t_minus_loss_list[0]),
                1 / (self.regulation_p + 1),
            )

        ## update parameters
        params = self.model.parameters()
        self.opt = self.optimizer_func(params, lr=self.policy_lr)
        self.opt.zero_grad()
        self.loss.backward()
        if self.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(params, self.max_gradient_norm)
        self.opt.step()

        total_norm = 0
        for p in params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.norm = total_norm

        ## Summarize
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Norm"] = self.norm
        print(f"Step {self.global_step}\tLoss {self.loss}\tNorm {self.norm}")
        self.global_step += 1

        return self.loss_summary, self.norm_summary

    def validation(self, input_feed):
        self.model.eval()
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.rank_list_size, self.device
        )
        with torch.no_grad():
            self.output = self.ranking_model()
        pad_removed_output = self.remove_padding_for_metric_eval()
        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        for metric in self.metric_type:
            topns = self.metric_topn
            metric_values = metrics.make_ranking_metric_fn(metric, topns)(
                self.labels, pad_removed_output, None
            )
            for topn, metric_value in zip(topns, metric_values):
                self.eval_summary[f"{metric}_{topn}"] = metric_value.item()

        return None, self.output, self.eval_summary  # no loss, outputs, summary.

    def remove_padding_for_metric_eval(self):
        output_scores = torch.unbind(self.output, dim=1)
        if len(output_scores) > len(self.docid_inputs):
            raise AssertionError(
                "Input id list is shorter than output score list when remove padding."
            )
        # Build mask
        valid_flags = torch.cat(
            (torch.ones(self.letor_features.shape[0]), torch.zeros([1])), dim=0
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
