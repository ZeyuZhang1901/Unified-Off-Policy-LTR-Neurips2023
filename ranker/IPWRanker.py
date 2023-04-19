import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import json

from ranker.AbstractRanker import AbstractRanker
from utils.input_feed import create_input_feed
from network.IPW import DNN
from utils import metrics


class IPWRanker(AbstractRanker):
    def __init__(
        self,
        hyper_json_file,
        feature_size,
        rank_list_size,
        max_visuable_size,
        click_model,
        propensity_estimator,  # propensity estimator
    ):
        with open(hyper_json_file) as ranker_json:
            hypers = json.load(ranker_json)
        self.policy_lr = hypers["policy_lr"]
        self.batch_size = hypers["batch_size"]
        self.l2_loss = hypers["l2_loss"]
        self.max_gradient_norm = hypers["max_gradient_norm"]
        self.metric_type = hypers["metric_type"]
        self.metric_topn = hypers["metric_topn"]
        self.objective_metric = hypers["objective_metric"]

        ## parameters from outside
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        self.click_model = click_model
        self.propensity_estimator = propensity_estimator

        self.model = DNN(feature_size=self.feature_size).to(
            self.device
        )  # ranking model

        self.loss_func = self.softmax_loss
        # self.loss_func = self.sigmoid_loss_on_list
        # self.loss_func = self.pairwise_loss_on_list
        self.optimizer_func = torch.optim.Adam  # optimizer

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

        ## Compute propensity weights for the input data
        if self.click_model.model_name == "dependent_click_model" or self.click_model.model_name ==  "cascade_model":
            lbd, clicks = [], []
        else:
            pw = []
        for i in range(len(input_feed[f"label0"])):
            click_list = [
                input_feed[f"label{l}"][i] for l in range(self.max_visuable_size)
            ]

            if self.click_model.model_name == "dependent_click_model" or self.click_model.model_name ==  "cascade_model":
                lbd_list = self.propensity_estimator.getLambdaForOneList(click_list)
                lbd.append(lbd_list)
                clicks.append(click_list)
            else:
                pw_list = self.propensity_estimator.getPropensityForOneList(click_list)
                pw.append(pw_list)

        ## train
        if self.click_model.model_name == "dependent_click_model" or self.click_model.model_name ==  "cascade_model":
            train_lbd = torch.as_tensor(np.array(lbd)).to(self.device)
            train_clicks = torch.as_tensor(np.array(clicks)).to(self.device)
            train_pw = 1 / (torch.cumprod(1+1e-9- train_clicks * (1 - (train_lbd+1e-4)), dim=1)
                / (1 +1e-9- train_clicks * (1 - (train_lbd + 1e-4))))
        else:
            train_pw = torch.as_tensor(np.array(pw)).to(self.device)

        train_output = self.ranking_model()
        train_labels = self.labels

        self.loss = self.loss_func(train_output, train_labels, train_pw)

        ## update
        self.separate_gradient_update()

        ## summary
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        print(f"Step {self.global_step}\tLoss {self.loss}\tGradient Norm {self.norm}")
        self.global_step += 1

        return self.loss_summary, self.norm_summary

    def separate_gradient_update(self):
        ranking_model_params = self.model.parameters()

        if self.l2_loss > 0:
            for p in ranking_model_params:
                self.loss += self.l2_loss * torch.sum(p**2) / 2

        opt_ranker = self.optimizer_func(self.model.parameters(), self.policy_lr)
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

    def softmax_loss(self, output, labels, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)
        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        loss = softmax_cross_entropy_with_logits(
            logits=output, labels=label_dis
        ) * torch.sum(weighted_labels, 1)
        return torch.sum(loss) / torch.sum(weighted_labels)

    def pairwise_cross_entropy_loss(
        self, pos_scores, neg_scores, propensity_weights=None
    ):
        """Computes pairwise softmax loss without propensity weighting.

        Args:
            pos_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a positive example.
            neg_scores: (torch.Tensor) A tensor with shape [batch_size, 1]. Each value is
            the ranking score of a negative example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(pos_scores)
        label_dis = torch.cat(
            [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1
        )
        loss = (
            softmax_cross_entropy_with_logits(
                logits=torch.cat([pos_scores, neg_scores], dim=1), labels=label_dis
            )
            * propensity_weights
        )
        return loss

    def sigmoid_loss_on_list(self, output, labels, propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        label_dis = torch.minimum(labels, 1)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(output, labels) * propensity_weights
        return torch.mean(torch.sum(loss, dim=1))

    def pairwise_loss_on_list(self, output, labels, propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, dim=1)
        sliced_label = torch.unbind(labels, dim=1)
        sliced_propensity = torch.unbind(propensity_weights, dim=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(sliced_label[i] - sliced_label[j])
                cur_propensity = (
                    sliced_propensity[i] * sliced_label[i]
                    + sliced_propensity[j] * sliced_label[j]
                )
                cur_pair_loss = -torch.exp(sliced_output[i]) / (
                    torch.exp(sliced_output[i]) + torch.exp(sliced_output[j])
                )
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.size()[0]
        return torch.sum(loss) / batch_size.type(torch.float32)


def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    loss = torch.sum(-labels * F.log_softmax(logits, -1), -1)
    return loss
