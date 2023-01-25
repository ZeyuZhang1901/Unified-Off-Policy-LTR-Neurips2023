import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from ranker.input_feed import create_input_feed
from utils import metrics


class RNNRanker(AbstractRanker):
    def __init__(
        self,
        feature_dim,
        click_model,
        learning_rate,
        rank_list_size,
        metric_type,
        metric_topn,
        objective_metric="ndcg_10",
        max_gradient_norm=5.0,
        batch_size=256,
        max_visuable_size=10,  # max length user can see
        l2_loss=0.0,  # for L2 regularization
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_dim = feature_dim
        self.l2_loss = l2_loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        self.max_gradient_norm = max_gradient_norm
        self.click_model = click_model
        self.model = nn.RNN(
            input_size=self.feature_dim,
            hidden_size=1,
            num_layers=3,
        ).to(self.device)
        self.optimizer_func = torch.optim.Adam
        # self.loss_func = F.mse_loss
        self.loss_func = self.softmax_loss

        self.metric_type = metric_type
        self.metric_topn = metric_topn
        self.objective_metric = objective_metric

        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.eval_summary = {}

    def arrange_batch(
        self,
        input_id_list,
    ):
        """Rearrange batch into (s,a,r,s') format."""

        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        reward_input_list = []
        for i in range(len(input_id_list)):
            ## actions
            input_feature_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[i], 0)).to(
                    torch.float32
                )
            )

            ## rewards
            reward_input_list.append(
                torch.index_select(self.labels.cpu(), dim=1, index=torch.tensor([i]))
            )

        ## All in CPU
        return (
            input_feature_list,
            reward_input_list,
        )

    def get_ranking_scores(
        self,
        input_feature_list,
    ):
        """Get ranking scores for each input rank list"""
        local_batch_size = input_feature_list[0].shape[0]

        input_features = (
            torch.cat(input_feature_list, dim=0)
            .reshape(local_batch_size, -1, self.feature_dim)
            .transpose(0, 1)
        ).to(self.device)
        return self.model(input_features)  # shape: [list_length, batch_size, 1]

    def update_policy(
        self,
        input_feed,
    ):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        input_feature_list, reward_input_list = self.arrange_batch(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )
        self.model.train()

        ## arrange reward with shape [list_length, batch_size, 1]
        # rewards = (
        #     torch.cat(reward_input_list, dim=1)
        #     .transpose(0, 1)
        #     .unsqueeze(-1)
        #     .to(self.device)
        # )
        rewards = torch.cat(reward_input_list, dim=1).to(self.device)
        ## get scores
        scores, _ = self.get_ranking_scores(input_feature_list)
        self.loss = self.loss_func(scores.transpose(0,1).squeeze(-1), rewards)

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
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(self.docid_inputs)):
            input_feature_list.append(
                torch.from_numpy(
                    np.take(self.letor_features, self.docid_inputs[i], 0)
                ).to(torch.float32)
            )

        local_batch_size = input_feature_list[0].shape[0]

        input_features = (
            torch.cat(input_feature_list, dim=0)
            .reshape(local_batch_size, -1, self.feature_dim)
            .transpose(0, 1)
        ).to(self.device)
        scores, _ = self.model(input_features)
        return scores.transpose(0, 1).squeeze(-1)

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
                torch.ones_like(output_scores[i], device=self.device) * -1e6,
            )
        return torch.stack(output_scores, dim=1)

    def softmax_loss(self, output, labels):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        weighted_labels = (labels + 1e-7)
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
