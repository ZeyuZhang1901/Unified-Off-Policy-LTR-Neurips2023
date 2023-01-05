import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from ranker.AbstractRanker import AbstractRanker
from network.DLA import DNN, DenoisingNet
from utils import metrics


class DLARanker(AbstractRanker):
    def __init__(
        self,
        feature_size,
        batch_size,
        learning_rate,
        click_model,
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        propensity_learning_rate=-1,  # The learning rate for ranker (-1 means same with learning_rate).
        rank_list_size=10,
        dynamic_bias_eta_change=0.0,  # Set eta change step for dynamic bias severity in training, 0.0 means no change
        dynamic_bias_step_interval=1000,  # Set how many steps to change eta for dynamic bias severity in training, 0.0 means no change
        max_propensity_weight=-1,  # Set maximum value for propensity weights, -1 means don't set
        l2_loss=0.0,  # Set strength for L2 regularization.
    ):
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.dynamic_bias_eta_change = dynamic_bias_eta_change
        self.dynamic_bias_step_interval = dynamic_bias_step_interval
        self.max_propensity_weight = max_propensity_weight
        self.l2_loss = l2_loss
        self.learning_rate = learning_rate
        self.propensity_learning_rate = (
            learning_rate if propensity_learning_rate < 0 else propensity_learning_rate
        )
        self.max_gradient_norm = max_gradient_norm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.click_model = click_model
        self.model = DNN(feature_size=self.feature_size).to(
            self.device
        )  # ranking model
        self.propensity_model = DenoisingNet(self.rank_list_size).to(
            self.device
        )  # propensity model
        self.logits_to_prob = nn.Softmax(dim=-1)  # logits to prob function
        self.loss_func = self.softmax_loss
        self.optimizer_func = torch.optim.Adam  # optimizer

        self.metric_type = ["mrr", "ndcg"]
        self.metric_topn = [3, 5, 10]
        self.objective_metric = "ndcg_10"

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
            input_id_list=self.docid_inputs[: self.rank_list_size]
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

    # %%
    # TODO Train step and its auxiliary functions

    def update_policy(self, input_feed):

        # train one epoch start
        self.create_input_feed(input_feed, self.rank_list_size)
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
        self.loss_summary["Total loss"] = self.loss
        self.norm_summary["Gradient Norm"] = self.norm
        print(
            f"Step {self.global_step}: Rank loss {self.rank_loss}\tExam loss {self.exam_loss}\tTotal loss {self.loss.item()}"
        )
        self.global_step += 1

        return self.loss.item(), self.loss_summary, self.norm_summary

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

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        r"""Clips gradient of an iterable of parameters at specified value.

        Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)

    # %%
    # TODO Validation step and its auxiliary functions

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.rank_list_size)
        with torch.no_grad():
            self.output = self.ranking_model()
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

    # %%
    # TODO Get train batch
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
        for i in range(local_batch_size):
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
    # TODO Get validation batch
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
        for index in range(len(qids)):
            # while len(docid_inputs) < len(qids):
            #     index = int(random.random() * len(qids))
            self.prepare_true_labels(
                dataset, index, docid_inputs, letor_features, labels, check_validation
            )
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)
        for i in range(local_batch_size):
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

    def get_all_query_result_list(self, dataset):
        pass

    def get_query_result_list(self, dataset, query):
        pass

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
