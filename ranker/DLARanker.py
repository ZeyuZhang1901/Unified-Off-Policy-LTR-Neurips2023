import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from ranker.AbstractRanker import AbstractRanker
from ranker.input_feed import create_input_feed
from network.DLA import DNN, DenoisingNet
from utils import metrics


class DLARanker(AbstractRanker):
    def __init__(
        self,
        feature_size,
        click_model,
        learning_rate,
        rank_list_size,
        metric_type,
        metric_topn,
        objective_metric="ndcg_10",
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        batch_size=256,
        max_visuable_size=10,
        propensity_learning_rate=-1,  # The learning rate for ranker (-1 means same with learning_rate).
        # dynamic_bias_eta_change=0.0,  # Set eta change step for dynamic bias severity in training, 0.0 means no change
        # dynamic_bias_step_interval=1000,  # Set how many steps to change eta for dynamic bias severity in training, 0.0 means no change
        max_propensity_weight=-1,  # Set maximum value for propensity weights, -1 means don't set
        l2_loss=0.0,  # Set strength for L2 regularization.
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        # self.dynamic_bias_eta_change = dynamic_bias_eta_change
        # self.dynamic_bias_step_interval = dynamic_bias_step_interval
        self.max_propensity_weight = max_propensity_weight
        self.l2_loss = l2_loss
        self.learning_rate = learning_rate
        self.propensity_learning_rate = (
            learning_rate if propensity_learning_rate < 0 else propensity_learning_rate
        )
        self.max_gradient_norm = max_gradient_norm
        
        self.click_model = click_model
        self.model = DNN(feature_size=self.feature_size).to(
            self.device
        )  # ranking model
        self.propensity_model = DenoisingNet(self.max_visuable_size).to(
            self.device
        )  # propensity model
        self.logits_to_prob = nn.Softmax(dim=-1)  # logits to prob function
        self.loss_func = self.softmax_loss
        self.optimizer_func = torch.optim.Adam  # optimizer

        self.metric_type = metric_type
        self.metric_topn = metric_topn
        self.objective_metric = objective_metric

        self.global_batch_count = 0
        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.eval_summary = {}

    # %%
    # TODO Model forward

    def ranking_model(self):
        """Construct ranking model

        Returns:
            A tensor with the same shape of input_docids.

        """
        output_scores = self.get_ranking_scores(
            input_id_list=self.docid_inputs
        )
        return torch.cat(output_scores, 1)

    def get_ranking_scores(
        self,
        input_id_list,
    ):
        """Compute ranking scores with the given inputs.

        Args:
            model: (BaseRankingModel) The model that is used to compute the ranking score.
            input_id_list: (list<torch.Tensor>) A list of tensors containing document ids.
                            Each tensor must have a shape of [None].
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A tensor with the same shape of input_docids.

        """
        # Build feature padding
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(letor_features, input_id_list[i], 0))
            )
        return self.model.forward(input_feature_list)

    def update_policy(self, input_feed):
        # train one epoch start
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.model.train()
        train_output = self.ranking_model()
        self.propensity_model.train()
        propensity_labels = torch.transpose(self.labels, 0, 1)
        self.propensity = self.propensity_model(propensity_labels)
        with torch.no_grad():
            self.propensity_weights = self.get_normalized_weights(
                self.logits_to_prob(self.propensity)
            )
        # rank_loss
        self.rank_loss = self.loss_func(
            train_output, self.labels, self.propensity_weights
        )
        # examination_loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(
                self.logits_to_prob(train_output)
            )
        self.exam_loss = self.loss_func(
            self.propensity, self.labels, self.relevance_weights
        )
        # total_loss
        self.loss = self.exam_loss + self.rank_loss

        # update
        self.separate_gradient_update()

        # summary
        self.loss_summary["Rank loss"] = torch.mean(self.rank_loss)
        self.loss_summary["Exam loss"] = torch.mean(self.exam_loss)
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        print(
            f"Step {self.global_step}: Rank Loss {self.rank_loss}\tExam Loss {self.exam_loss}\tTotal Loss {self.loss.item()}"
        )
        self.global_step += 1

        return self.loss_summary, self.norm_summary

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.

        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
                shape=[batch_size * rank_list_size]

        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.max_propensity_weight > 0:
            self.clip_grad_value(
                propensity_weights,
                clip_value_min=0,
                clip_value_max=self.max_propensity_weight,
            )
        return propensity_weights

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.l2_loss * torch.sum(p**2) / 2
        self.loss = self.exam_loss + self.rank_loss

        opt_denoise = self.optimizer_func(
            self.propensity_model.parameters(), self.propensity_learning_rate
        )
        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)

        opt_denoise.zero_grad()
        opt_ranker.zero_grad()

        self.loss.backward()

        if self.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(
                self.propensity_model.parameters(), self.max_gradient_norm
            )
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)

        opt_denoise.step()
        opt_ranker.step()

        total_norm = 0

        for p in denoise_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
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
