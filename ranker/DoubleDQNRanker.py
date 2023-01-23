import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from ranker.AbstractRanker import AbstractRanker
from ranker.input_feed import create_input_feed
from network.DoubleDQN import DoubleDQN
from utils import metrics


class DoubleDQNRanker(AbstractRanker):
    def __init__(
        self,
        feature_dim,
        click_model,  # click model used to generate clicks when constructing batch
        learning_rate,
        rank_list_size,
        metric_type,
        metric_topn,
        objective_metric="ndcg_10",
        target_update_step=50,  # target model update every ~ steps
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        batch_size=256,
        discount=0.9,
        max_visuable_size=10,  # user visuable length of rank list, usually 10
        # dynamic_bias_eta_change=0.0,  # Set eta change step for dynamic bias severity in training, 0.0 means no change
        # dynamic_bias_step_interval=1000,  # Set how many steps to change eta for dynamic bias severity in training, 0.0 means no change
        l2_loss=0.0,  # Set strength for L2 regularization.
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_dim = feature_dim
        self.l2_loss = l2_loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.discount = discount
        self.target_update_step = target_update_step
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        # self.dynamic_bias_eta_change = dynamic_bias_eta_change
        # self.dynamic_bias_step_interval = dynamic_bias_step_interval
        self.max_gradient_norm = max_gradient_norm
        self.click_model = click_model
        self.model = DoubleDQN(self.feature_dim, self.rank_list_size).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer_func = torch.optim.Adam
        self.loss_func = F.mse_loss

        self.metric_type = metric_type
        self.metric_topn = metric_topn
        self.objective_metric = objective_metric

        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.eval_summary = {}
        self.q_summary = {}

    def get_current_scores(
        self,
        input_id_list,
    ):

        # Build feature padding
        local_batch_size = input_id_list.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[i], 0)).to(
                    self.device
                )
            )
            # position_input_list.append(
            #     F.one_hot(
            #         torch.ones(local_batch_size, dtype=torch.int64) * i,
            #         self.rank_list_size,
            #     ).to(self.device)
            # )
            # position_input_list.append(
            #     (i * torch.ones(local_batch_size, 1)).to(self.device)
            # )
            position_input_list.append(  ## position-embedding
                torch.repeat_interleave(
                    torch.sin(
                        i
                        / torch.pow(
                            torch.tensor(10000),
                            torch.arange(self.feature_dim) * 2 / self.feature_dim,
                        )
                    ).reshape(1, -1),
                    local_batch_size,
                    dim=0,
                ).to(self.device)
            )
            # mask = torch.cat(
            #     [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            # ).to(self.device)
            # reward_input_list.append(
            #     torch.where(
            #         self.labels * torch.stack([mask] * local_batch_size) > 0,
            #         torch.ones_like(self.labels, dtype=torch.int),
            #         torch.zeros_like(self.labels, dtype=torch.int),
            #     )
            # )
        # for i in range(len(input_id_list)):
        #     if i == 0:
        #         cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
        #     else:
        #         cum_input_feature_list.append(
        #             torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
        #         )
        return self.model.forward_current(
            input_feature_list,
            # cum_input_feature_list,
            position_input_list,
            # reward_input_list,
        )

    def get_next_scores_and_indexs(
        self,
        input_id_list,
    ):

        local_batch_size = input_id_list.shape[1]
        # PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        # self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        candidates_list = []

        ## Action feature, position feature to form state
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[i], 0)).to(
                    self.device
                )
            )
            # position_input_list.append(
            #     F.one_hot(
            #         torch.ones(local_batch_size, dtype=torch.int64) * i,
            #         self.rank_list_size,
            #     )
            # )
            # position_input_list.append(  ## singal position
            #     (i * torch.ones(local_batch_size, 1)).to(self.device)
            # )
            position_input_list.append(  ## position-embedding
                torch.repeat_interleave(
                    torch.sin(
                        i
                        / torch.pow(
                            torch.tensor(10000),
                            torch.arange(self.feature_dim) * 2 / self.feature_dim,
                        )
                    ).reshape(1, -1),
                    local_batch_size,
                    dim=0,
                ).to(self.device)
            )

            # mask = torch.cat(
            #     [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            # ).to(self.device)
            # reward_input_list.append(
            #     torch.where(
            #         self.labels * torch.stack([mask] * local_batch_size) > 0,
            #         torch.ones_like(self.labels, dtype=torch.int),
            #         torch.zeros_like(self.labels, dtype=torch.int),
            #     )
            # )

        ## form candidate list and its valid mask (shape: [batch_size, max_visuable_size])
        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(
                    np.take(self.letor_features, input_id_list[:, i], 0)
                ).to(self.device)
            )

        candidate_num = candidates_list[0].shape[0]
        valid_flags = torch.cat(
            (torch.ones(self.letor_features.shape[0] - 1), torch.zeros([1])), dim=0
        ).type(torch.bool)
        valid_mask = torch.ones(local_batch_size, candidate_num, dtype=torch.bool).to(
            self.device
        )
        for i in range(candidate_num):
            valid_mask[:, i] = torch.index_select(
                input=valid_flags, dim=0, index=self.docid_inputs[i]
            )

        ## form cumulative features for state
        # for i in range(len(input_id_list)):
        #     if i == 0:
        #         cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
        #     else:
        #         cum_input_feature_list.append(
        #             torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
        #         )

        ## Get next scores and indexes
        return self.model.forward_next(
            valid_mask,
            # cum_input_feature_list,
            position_input_list,
            # reward_input_list,
            candidates_list,
        )  # return two lists of [batch_size * 1] tensors, each with rank_size items.

    def get_target_values(self, input_index_list):
        local_batch_size = self.docid_inputs.shape[1]
        # PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        # letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        docid_list, input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        # cum_input_feature_list = []
        for i in range(len(input_index_list)):
            docid_list.append(
                torch.gather(
                    self.docid_inputs,
                    dim=0,
                    index=input_index_list[i].cpu().reshape(1, -1),
                )
            )
            input_feature_list.append(  # batch_size * feature_dim
                torch.from_numpy(
                    np.take(self.letor_features, docid_list[-1].flatten(), 0)
                ).to(self.device)
            )
            # position_input_list.append(
            #     F.one_hot(
            #         torch.ones(local_batch_size, dtype=torch.int64) * i,
            #         self.rank_list_size,
            #     )
            # )
            # position_input_list.append(  ## singal position
            #     (i * torch.ones(local_batch_size, 1)).to(self.device)
            # )
            position_input_list.append(  ## position-embedding
                torch.repeat_interleave(
                    torch.sin(
                        i
                        / torch.pow(
                            torch.tensor(10000),
                            torch.arange(self.feature_dim) * 2 / self.feature_dim,
                        )
                    ).reshape(1, -1),
                    local_batch_size,
                    dim=0,
                ).to(self.device)
            )

            # mask = torch.cat(
            #     [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            # ).to(self.device)
            # reward_input_list.append(
            #     torch.where(
            #         self.labels * torch.stack([mask] * local_batch_size) > 0,
            #         torch.ones_like(self.labels, dtype=torch.int),
            #         torch.zeros_like(self.labels, dtype=torch.int),
            #     )
            # )
        # for i in range(len(input_index_list)):
        #     if i == 0:
        #         cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
        #     else:
        #         cum_input_feature_list.append(
        #             torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
        #         )
        return self.target_model.forward_current(
            input_feature_list,
            # cum_input_feature_list,
            position_input_list,
            # reward_input_list,
        )

    # %%
    # TODO Train step

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.model.train()
        self.target_model.train()
        current_scores_list = self.get_current_scores(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )  # list of `rank_size` tensors with shape [batch_size, 1]
        _, next_index_list = self.get_next_scores_and_indexs(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )  # two lists of `rank_size` tensors, each with shape [batch_size, 1]
        target_values_list = self.get_target_values(
            input_index_list=next_index_list
        )  # list of `rank_size` tensors with shape [batch_size, 1]

        target_list = []
        for i in range(len(target_values_list)):
            if i == len(target_values_list) - 1:
                target_list.append(
                    torch.index_select(
                        self.labels, dim=1, index=torch.tensor([i]).to(self.device)
                    )
                )
            else:
                target_list.append(
                    torch.index_select(
                        self.labels, dim=1, index=torch.tensor([i]).to(self.device)
                    )
                    + self.discount * target_values_list[i + 1]
                )
        self.loss = self.loss_func(
            torch.cat(current_scores_list, dim=1), torch.cat(target_list, dim=1)
        )
        self.policy_q = torch.cat(current_scores_list, dim=0).mean()
        self.target_q = torch.cat(target_list, dim=0).mean()

        # update
        self.separate_gradient_update()

        # summary
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        self.q_summary["Policy_Q"] = self.policy_q
        self.q_summary["Target_Q"] = self.target_q
        print(
            f"Step {self.global_step}: Loss {self.loss}\tGradient Norm {self.norm}\tPolicy Q {self.policy_q}\tTarget Q {self.target_q}"
        )
        self.global_step += 1

        return self.loss.item(), self.loss_summary, self.norm_summary, self.q_summary

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

        if self.global_step % self.target_update_step == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # %%
    # TODO Validation step and its auxiliary functions
    def validation(self, input_feed):
        self.model.eval()
        self.target_model.eval()
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.rank_list_size, self.device
        )
        with torch.no_grad():
            self.output = self.validation_forward()
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

    def validation_forward(self):
        local_batch_size = self.docid_inputs.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)

        # input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        candidates_list = []

        indices = torch.zeros(
            local_batch_size, self.rank_list_size, dtype=torch.int64, device=self.device
        )
        # labels = torch.zeros(
        #     local_batch_size, self.rank_list_size, dtype=torch.int64, device=self.device
        # )
        masks = torch.ones(
            local_batch_size, self.rank_list_size, dtype=torch.bool, device=self.device
        )
        docid_list = []

        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(
                    np.take(self.letor_features, self.docid_inputs[:, i], 0)
                ).to(self.device)
            )
        for i in range(self.rank_list_size):  # for each rank position
            # if i == 0:
            #     cum_input_feature_list.append(
            #         torch.zeros(local_batch_size, self.feature_dim)
            #     )
            # else:
            #     cum_input_feature_list.append(
            #         sum(input_feature_list) / len(input_feature_list)
            #     )
            # position_input_list.append(
            #     F.one_hot(
            #         torch.ones(local_batch_size, dtype=torch.int64) * i,
            #         self.rank_list_size,
            #     )
            # )
            # position_input_list.append(
            #     (i * torch.ones(local_batch_size, 1)).to(self.device)
            # )
            position_input_list.append(  ## position-embedding
                torch.repeat_interleave(
                    torch.sin(
                        i
                        / torch.pow(
                            torch.tensor(10000),
                            torch.arange(self.feature_dim) * 2 / self.feature_dim,
                        )
                    ).reshape(1, -1),
                    local_batch_size,
                    dim=0,
                ).to(self.device)
            )

            # if i == 0:
            #     reward_input_list.append(
            #         torch.zeros(local_batch_size, self.rank_list_size).to(self.device)
            #     )
            # else:
            #     mask = torch.cat(
            #         [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            #     ).to(self.device)
            #     reward_input_list.append(
            #         torch.where(
            #             labels * torch.stack([mask] * local_batch_size) > 0,
            #             torch.ones_like(self.labels, dtype=torch.int),
            #             torch.zeros_like(self.labels, dtype=torch.int),
            #         )
            #     )
            _, index = self.model.forward(
                masks,
                # cum_input_feature_list,
                position_input_list,
                # reward_input_list,
                candidates_list,
            )
            indices[:, i] = index
            docid_list.append(  # list of [1 * batch_size] tensors
                torch.gather(self.docid_inputs, dim=0, index=index.cpu().reshape(1, -1))
            )
            # input_feature_list.append(  # batch_size * feature_dim
            #     torch.from_numpy(
            #         np.take(self.letor_features, docid_list[-1].flatten(), 0)
            #     ).to(self.device)
            # )
            # labels[:, i] = torch.gather(
            #     self.labels, dim=1, index=index.reshape(-1, 1)
            # ).flatten()
            masks[torch.arange(local_batch_size), index] = False

        output = torch.zeros(
            local_batch_size,
            self.rank_list_size,
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(self.rank_list_size):
            output[torch.arange(local_batch_size), indices[:, i]] = (
                self.rank_list_size - i
            )

        return output

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
