import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from ranker.AbstractRanker import AbstractRanker
from network.DQN import DQN
from utils import metrics


class DQNRanker(AbstractRanker):
    def __init__(
        self,
        feature_dim,
        click_model,  # click model used to generate clicks when constructing batch
        learning_rate,
        target_update_step=50,  # target model update every ~ steps
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        batch_size=256,
        discount=0.9,
        rank_list_size=10,  # considered length of each rank list, usually 10
        dynamic_bias_eta_change=0.0,  # Set eta change step for dynamic bias severity in training, 0.0 means no change
        dynamic_bias_step_interval=1000,  # Set how many steps to change eta for dynamic bias severity in training, 0.0 means no change
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
        self.dynamic_bias_eta_change = dynamic_bias_eta_change
        self.dynamic_bias_step_interval = dynamic_bias_step_interval
        self.max_gradient_norm = max_gradient_norm
        self.click_model = click_model
        self.model = DQN(self.feature_dim, self.rank_list_size).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer_func = torch.optim.Adam
        self.loss_func = F.mse_loss

        self.metric_type = ["mrr", "ndcg"]
        self.metric_topn = [3, 5, 10]
        self.objective_metric = "ndcg_10"

        self.global_batch_count = 0
        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.eval_summary = {}

    # %%
    # TODO get train batch

    def prepare_clicks_for_one_list(
        self,
        dataset,
        index,
        docid_input,
        letor_features,
        labels,
        check_validation=True,
    ):
        # Generate clicks with click models.
        qids = dataset.get_all_querys()
        label_list = dataset.get_all_relevance_label_by_query(qids[index])
        if len(label_list) < self.rank_list_size:
            label_list += [0] * (self.rank_list_size - len(label_list))
        label_list = label_list[: self.rank_list_size]
        click_list, _, _ = self.click_model.sampleClicksForOneList(list(label_list))

        # Check if data is valid
        if check_validation:
            if sum(click_list) == 0:
                return
        base = len(letor_features)
        initial_list = dataset.get_candidate_docids_by_query(qids[index])[
            : self.rank_list_size
        ]
        for x in range(self.rank_list_size):
            if initial_list[x] >= 0:
                letor_features.append(
                    dataset.get_features_by_query_and_docid(qids[index], x)
                )
        docid_input.append(
            list(
                [
                    -1 if initial_list[x] < 0 else base + x
                    for x in range(self.rank_list_size)
                ]
            )
        )
        labels.append(click_list)

    def get_train_batch(
        self,
        dataset,
        check_validation=True,
    ):
        """Get a random batch of data, prepared for step. Typically used for training
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).
        """

        qids = dataset.get_all_querys()  # nparray of qids
        docid_inputs, letor_features, labels = [], [], []
        rank_list_idxs = []
        batch_num = len(docid_inputs)
        while len(docid_inputs) < self.batch_size:
            index = int(random.random() * len(qids))
            self.prepare_clicks_for_one_list(
                dataset, index, docid_inputs, letor_features, labels, check_validation
            )
            if batch_num < len(docid_inputs):
                rank_list_idxs.append(index)
                batch_num = len(docid_inputs)
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(
            local_batch_size
        ):  # if doc doesn't exist, use zeros array as feature vector
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array(
                    [
                        docid_inputs[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array(
                    [
                        labels[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
        # Create input feed map
        input_feed = {}
        input_feed["letor_features"] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[f"docid_input{l}"] = batch_docid_inputs[l]
            input_feed[f"label{l}"] = batch_labels[l]
        # Create info_map to store other information
        info_map = {
            "rank_list_idxs": rank_list_idxs,
            "input_list": docid_inputs,
            "click_list": labels,
            "letor_features": letor_features,
        }

        self.global_batch_count += 1
        if self.dynamic_bias_eta_change != 0:
            if self.global_batch_count % self.dynamic_bias_step_interval == 0:
                self.click_model.eta += self.dynamic_bias_eta_change
                self.click_model.setExamProb(self.click_model.eta)
                print(
                    "Dynamically change bias severity eta to %.3f"
                    % self.click_model.eta
                )

        return input_feed, info_map

    # %%
    # TODO get validation batch

    def prepare_true_labels(
        self,
        dataset,
        index,
        docid_inputs,
        letor_features,
        labels,
        check_validation=True,
    ):
        qids = dataset.get_all_querys()
        label_list = dataset.get_all_relevance_label_by_query(qids[index])
        if len(label_list) < self.rank_list_size:
            label_list += [0] * (self.rank_list_size - len(label_list))
        label_list = label_list[: self.rank_list_size]

        # Check if data is valid
        if check_validation and sum(label_list) == 0:
            return
        base = len(letor_features)
        initial_list = dataset.get_candidate_docids_by_query(qids[index])[
            : self.rank_list_size
        ]
        for x in range(self.rank_list_size):
            if initial_list[x] >= 0:
                letor_features.append(
                    dataset.get_features_by_query_and_docid(qids[index], x)
                )
        docid_inputs.append(
            list(
                [
                    -1 if initial_list[x] < 0 else base + x
                    for x in range(self.rank_list_size)
                ]
            )
        )
        labels.append(label_list)
        return

    def get_validation_batch(
        self,
        dataset,
        check_validation=True,
    ):
        """Get whole batch of data, prepared for step. Typically used for validation
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            check_validation: (bool) Set True to ignore data with no positive labels.

        Returns:
            input_feed: a feed dictionary for the next step
            info_map: a dictionary contain some basic information about the batch (for debugging).
        """

        qids = dataset.get_all_querys()  # nparray of qids
        docid_inputs, letor_features, labels = [], [], []
        for index in range(len(qids)):  # for each query
            # while len(docid_inputs) < len(qids):
            #     index = int(random.random() * len(qids))
            self.prepare_true_labels(
                dataset, index, docid_inputs, letor_features, labels, check_validation
            )
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(
            local_batch_size
        ):  # if doc doesn't exist, use zeros array as feature vector
            for j in range(self.rank_list_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.rank_list_size):
            # Batch encoder inputs are just re-indexed docid_inputs.
            batch_docid_inputs.append(
                np.array(
                    [
                        docid_inputs[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
            # Batch decoder inputs are re-indexed decoder_inputs, we create
            # labels.
            batch_labels.append(
                np.array(
                    [
                        labels[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
        # Create input feed map
        input_feed = {}
        input_feed["letor_features"] = np.array(letor_features)
        for l in range(self.rank_list_size):
            input_feed[f"docid_input{l}"] = batch_docid_inputs[l]
            input_feed[f"label{l}"] = batch_labels[l]
        # Create info_map to store other information
        info_map = {
            "input_list": docid_inputs,
            "click_list": labels,
        }

        return input_feed, info_map

    # %%
    # TODO Create the input from input_feed to run the model

    def create_input_feed(self, input_feed, list_size):
        """Create the input from input_feed to run the model

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.
            list_size: (int) The top number of documents to consider in the input docids.
        """
        self.labels = []
        self.docid_inputs = []
        self.letor_features = input_feed["letor_features"]
        for i in range(list_size):
            self.docid_inputs.append(input_feed[f"docid_input{i}"])
            self.labels.append(input_feed[f"label{i}"])
        self.labels = np.transpose(self.labels)
        self.labels = torch.from_numpy(self.labels).to(self.device)
        self.docid_inputs = torch.as_tensor(data=self.docid_inputs, dtype=torch.int64)

    # %%
    # TODO Run Model
    def get_current_scores(
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
            list of `rank_size` tensors (scores), with shape [batch_size, 1]

        """
        # Build feature padding
        local_batch_size = input_id_list.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(letor_features, input_id_list[i], 0)).to(
                    self.device
                )
            )
            position_input_list.append(
                F.one_hot(
                    torch.ones(local_batch_size, dtype=torch.int64) * i,
                    self.rank_list_size,
                ).to(self.device)
            )
            mask = torch.cat(
                [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            ).to(self.device)
            # reward_input_list.append(
            #     torch.where(
            #         self.labels * torch.stack([mask] * local_batch_size) > 0,
            #         torch.ones_like(self.labels, dtype=torch.int),
            #         torch.zeros_like(self.labels, dtype=torch.int),
            #     )
            # )
        for i in range(len(input_id_list)):
            if i == 0:
                cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
            else:
                cum_input_feature_list.append(
                    torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
                )
        return self.model.forward_current(
            input_feature_list,
            cum_input_feature_list,
            position_input_list,
            # reward_input_list,
        )

    def get_next_scores(
        self,
        input_id_list,
    ):
        local_batch_size = input_id_list.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        candidates_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(
                torch.from_numpy(np.take(letor_features, input_id_list[i], 0)).to(
                    self.device
                )
            )
            position_input_list.append(
                F.one_hot(
                    torch.ones(local_batch_size, dtype=torch.int64) * i,
                    self.rank_list_size,
                ).to(self.device)
            )
            mask = torch.cat(
                [torch.ones(i), torch.zeros(self.rank_list_size - i)], dim=0
            ).to(self.device)
            # reward_input_list.append(
            #     torch.where(
            #         self.labels * torch.stack([mask] * local_batch_size) > 0,
            #         torch.ones_like(self.labels, dtype=torch.int),
            #         torch.zeros_like(self.labels, dtype=torch.int),
            #     )
            # )
        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(np.take(letor_features, input_id_list[:, i], 0)).to(
                    self.device
                )
            )
        for i in range(len(input_id_list)):
            if i == 0:
                cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
            else:
                cum_input_feature_list.append(
                    torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
                )
        return self.target_model.forward_next(
            cum_input_feature_list,
            position_input_list,
            # reward_input_list,
            candidates_list,
        )

    # %%
    # TODO Train step

    def update_policy(self, input_feed):
        self.create_input_feed(input_feed, self.rank_list_size)
        self.model.train()
        self.target_model.train()
        current_scores_list = self.get_current_scores(
            input_id_list=self.docid_inputs[: self.rank_list_size]
        )  # list of `rank_size` tensors with shape [batch_size, 1]
        next_scores_list = self.get_next_scores(
            input_id_list=self.docid_inputs[: self.rank_list_size]
        )  # list of `rank_size` tensors with shape [batch_size, 1]
        target_list = []
        for i in range(len(next_scores_list)):
            if i == len(next_scores_list) - 1:
                target_list.append(
                    torch.index_select(
                        self.labels, dim=1, index=torch.tensor([i]).to(self.device)
                    )
                )
            else:
                target_list.append(
                    torch.index_select(self.labels, dim=1, index=torch.tensor([i]).to(self.device))
                    + self.discount * next_scores_list[i + 1]
                )
        self.loss = self.loss_func(
            torch.cat(current_scores_list, dim=1), torch.cat(target_list, dim=1)
        )

        # update
        self.separate_gradient_update()

        # summary
        self.loss_summary["Loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        print(f"Step {self.global_step}: Loss {self.loss}\tGradient Norm {self.norm}")
        self.global_step += 1

        return self.loss.item(), self.loss_summary, self.norm_summary

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
    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.rank_list_size)
        with torch.no_grad():
            self.output = self.validation_forward()
        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output
            )
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
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)

        input_feature_list, cum_input_feature_list = [], []
        # position_input_list, reward_input_list = [], []
        position_input_list = []
        candidates_list = []

        indices = torch.zeros(
            local_batch_size, self.rank_list_size, dtype=torch.int64, device=self.device
        )
        labels = torch.zeros(
            local_batch_size, self.rank_list_size, dtype=torch.int64, device=self.device
        )
        masks = torch.ones(
            local_batch_size, self.rank_list_size, dtype=torch.bool, device=self.device
        )
        docid_list = []

        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(
                    np.take(letor_features, self.docid_inputs[:, i], 0)
                ).to(self.device)
            )
        for i in range(self.rank_list_size):  # for each rank position
            if i == 0:
                cum_input_feature_list.append(
                    torch.zeros(local_batch_size, self.feature_dim).to(self.device)
                )
            else:
                cum_input_feature_list.append(
                    sum(input_feature_list) / len(input_feature_list)
                )
            position_input_list.append(
                F.one_hot(
                    torch.ones(local_batch_size, dtype=torch.int64) * i,
                    self.rank_list_size,
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
                cum_input_feature_list,
                position_input_list,
                # reward_input_list,
                candidates_list,
            )
            indices[:, i] = index
            docid_list.append(  # list of [1 * batch_size] tensors
                torch.gather(self.docid_inputs, dim=0, index=index.cpu().reshape(1, -1))
            )
            input_feature_list.append(  # batch_size * feature_dim
                torch.from_numpy(
                    np.take(letor_features, docid_list[-1].flatten(), 0)
                ).to(self.device)
            )
            labels[:, i] = torch.gather(
                self.labels, dim=1, index=index.reshape(-1, 1)
            ).flatten()
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

    def remove_padding_for_metric_eval(self, input_id_list, model_output):
        output_scores = torch.unbind(model_output, dim=1)
        if len(output_scores) > len(input_id_list):
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
                input=valid_flags, dim=0, index=input_id_list[i]
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

    # def update_policy(self, memory, dataset):
    #     trainsitions = memory.sample(self.batch_size)
    #     batch = Transition(*zip(*trainsitions))
    #     states = torch.cat(batch.state).to(self.device)
    #     actions = torch.cat(batch.action).to(self.device)
    #     nextstates = torch.cat(batch.next_state).to(self.device)
    #     rewards = torch.cat(batch.reward).to(self.device)
    #     dones = torch.cat(batch.done).to(self.device)

    #     Q_s_a = self.model(states, actions)
    #     with torch.no_grad():
    #         Q_nexts_nexta = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(
    #             self.device
    #         )
    #         for i in range(self.batch_size):
    #             candidates = (
    #                 torch.tensor(dataset.get_all_features_by_query(batch.qid[i]))
    #                 .to(self.device)
    #                 .to(torch.float32)
    #             )
    #             candidates = candidates[batch.chosen[i]]
    #             Q_nexts_nexta[i, 0] = self.getTargetMaxValue(nextstates[i], candidates)
    #         Q_nexts_nexta = rewards + (1 - dones) * self.discount * Q_nexts_nexta

    #     loss = F.mse_loss(Q_s_a, Q_nexts_nexta)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # target network soft update
    #     for param, target_param in zip(
    #         self.model.parameters(), self.target_q.parameters()
    #     ):
    #         target_param.data.copy_(
    #             self.tau * param.data + (1 - self.tau) * target_param.data
    #         )

    #     return Q_s_a.mean().item(), Q_nexts_nexta.mean().item(), loss.item()

    # def get_query_result_list(self, dataset, query):
    #     candidates = dataset.get_all_features_by_query(query).astype(np.float32)
    #     docid_list = dataset.get_candidate_docids_by_query(query)
    #     end_pos = int((self.state_dim - self.action_dim) / 2)
    #     ndoc = len(docid_list) if len(docid_list) < end_pos else end_pos
    #     ranklist = np.zeros(ndoc, dtype=np.int32)

    #     state = np.zeros(self.state_dim, dtype=np.float32)
    #     next_state = np.zeros(self.state_dim, dtype=np.float32)
    #     for pos in range(ndoc):
    #         # state
    #         state = next_state
    #         # action
    #         action = self.selectAction(
    #             state=state.reshape(1, -1), candidates=candidates
    #         )
    #         # reward
    #         docid = dataset.get_docid_by_query_and_feature(query, action)
    #         relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
    #         ranklist[pos] = docid
    #         # next state
    #         # next_state[: self.action_dim] = action + pos * state[: self.action_dim] / (
    #         #     pos + 1
    #         # )
    #         # if pos < end_pos:
    #         #     next_state[self.action_dim + pos] = (
    #         #         1 / np.log2(pos + 2) if relevance > 0 else 0
    #         #     )
    #         #     next_state[self.action_dim + end_pos + pos] = 1
    #         #     if pos > 0:
    #         #         next_state[self.action_dim + end_pos + pos - 1] = 0
    #         # delete chosen doc in candidates
    #         for i in range(candidates.shape[0]):
    #             if np.array_equal(candidates[i], action):
    #                 candidates = np.delete(candidates, i, axis=0)
    #                 break

    #     return ranklist

    # def get_all_query_result_list(self, dataset):
    #     query_result_list = {}
    #     for query in dataset.get_all_querys():
    #         query_result_list[query] = self.get_query_result_list(dataset, query)

    #     return query_result_list

    # def getTargetMaxValue(self, state, candidates):

    #     scores = self.target_q(state.expand(candidates.shape[0], -1), candidates)
    #     return torch.max(scores, 0)[0].item()

    # def selectAction(self, state, candidates):
    #     with torch.no_grad():
    #         if type(state) == np.ndarray:
    #             state = torch.from_numpy(state)
    #         if type(candidates) == np.ndarray:
    #             candidates = torch.from_numpy(candidates)
    #         state = state.expand(candidates.shape[0], -1).to(self.device)
    #         candidates = candidates.to(self.device)
    #         scores = self.q(state, candidates)
    #         return candidates[torch.max(scores, 0)[1]].squeeze(0).cpu().numpy()

    # def restore_ranker(self, path):
    #     print("restore ranker start!")
    #     torch.save(self.q.state_dict(), path + "q.pt")
    #     torch.save(self.target_q.state_dict(), path + "target_q.pt")
    #     print("restore ranker finish!")

    # def load_ranker(self, path):
    #     print("load ranker start!")
    #     self.q.load_state_dict(torch.load(path + "q.pt"))
    #     self.target_q.load_state_dict(torch.load(path + "target_q.pt"))
    #     print("load ranker finish!")
