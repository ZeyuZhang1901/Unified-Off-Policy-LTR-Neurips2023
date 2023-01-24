import torch
import numpy as np
import random


"""Get train batch and validation batch"""

class Train_Input_feed(object):
    def __init__(
        self,
        click_model,
        max_visuable_size,
        batch_size,
    ) -> None:
        self.max_visuable_size = max_visuable_size  # max position the user can see
        self.batch_size = batch_size

        self.click_model = click_model

    def prepare_clicks_for_one_list(
        self,
        dataset,
        index,
        docid_input,
        letor_features,
        labels,
        check_validation=True,
    ):
        ## Get all labels for docs under this query (if invalid doc, set 0)
        ## Then simulate clicks for top `max_visuable_size` docs
        i = index
        label_list = [
            0 if dataset.initial_list[i][x] < 0 else dataset.labels[i][x]
            for x in range(self.max_visuable_size)
        ]
        click_list, _, _ = self.click_model.sampleClicksForOneList(list(label_list))

        ## Check if there is clicks in this query
        if check_validation and sum(click_list) == 0:
            return

        ## Get all seen doc_features under this query (if invalid doc, do nothing)
        ## Add features, docid_list, label_list to whole input list
        base = len(letor_features)
        for x in range(self.max_visuable_size):
            if dataset.initial_list[i][x] >= 0:
                letor_features.append(dataset.features[dataset.initial_list[i][x]])
        docid_input.append(
            list(
                [
                    -1 if dataset.initial_list[i][x] < 0 else base + x
                    for x in range(self.max_visuable_size)
                ]
            )
        )
        labels.append(click_list)

    def get_train_batch(
        self,
        dataset,
        check_validation=True,
    ):

        length = len(dataset.initial_list)
        docid_inputs, letor_features, labels = [], [], []
        rank_list_idxs = []
        batch_num = len(docid_inputs)

        ## prepare docids, features and clicks for sampled queries
        ## total `batch_size` sampled queries
        while len(docid_inputs) < self.batch_size:
            index = int(random.random() * length)
            self.prepare_clicks_for_one_list(
                dataset, index, docid_inputs, letor_features, labels, check_validation
            )
            if batch_num < len(docid_inputs):
                rank_list_idxs.append(index)
                batch_num = len(docid_inputs)
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)

        ## mask invalid docids with max index (refer to zero feature vector)
        for i in range(local_batch_size):
            for j in range(self.max_visuable_size):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        ## construct batch_docid_inputs (list of float32 nparrays with shape [batch_size])
        ## and batch_labels (list of float32 nparrays with shape [batch_size])
        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_visuable_size):
            batch_docid_inputs.append(
                np.array(
                    [
                        docid_inputs[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
            batch_labels.append(
                np.array(
                    [
                        labels[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )

        ## Create input feed map
        input_feed = {}
        input_feed["letor_features"] = np.array(letor_features)
        for l in range(self.max_visuable_size):
            input_feed[f"docid_input{l}"] = batch_docid_inputs[l]
            input_feed[f"label{l}"] = batch_labels[l]

        # self.global_batch_count += 1
        # if self.dynamic_bias_eta_change != 0:
        #     if self.global_batch_count % self.dynamic_bias_step_interval == 0:
        #         self.click_model.eta += self.dynamic_bias_eta_change
        #         self.click_model.setExamProb(self.click_model.eta)
        #         print(
        #             "Dynamically change bias severity eta to %.3f"
        #             % self.click_model.eta
        #         )

        return input_feed


class Validation_Input_feed(object):
    def __init__(
        self,
        max_candidate_num,
    ) -> None:
        self.max_candidate_num = max_candidate_num

    def prepare_true_labels(
        self,
        dataset,
        index,
        docid_inputs,
        letor_features,
        labels,
        check_validation=False,
    ):
        ## Get all labels for docs under this query (if invalid doc, set 0)
        i = index
        label_list = [
            0 if dataset.initial_list[i][x] < 0 else dataset.labels[i][x]
            for x in range(self.max_candidate_num)
        ]

        ## Check if there is relevant docs in this query
        if check_validation and sum(label_list) == 0:
            return

        ## Get all doc_features under this query (if invalid doc, do nothing)
        ## Add  features, docid_list, label_list to whole input list
        base = len(letor_features)
        for x in range(self.max_candidate_num):
            if dataset.initial_list[i][x] >= 0:
                letor_features.append(dataset.features[dataset.initial_list[i][x]])
        docid_inputs.append(
            list(
                [
                    -1 if dataset.initial_list[i][x] < 0 else base + x
                    for x in range(self.max_candidate_num)
                ]
            )
        )
        labels.append(label_list)
        return

    def get_validation_batch(
        self,
        dataset,
        check_validation=False,
    ):

        length = len(dataset.initial_list)
        docid_inputs, letor_features, labels = [], [], []

        ## prepare docids, features and labels for each query
        for index in range(length):
            self.prepare_true_labels(
                dataset, index, docid_inputs, letor_features, labels, check_validation
            )
        local_batch_size = len(docid_inputs)
        letor_features_length = len(letor_features)

        ## mask invalid docids with max index (refer to zero feature vector)
        for i in range(local_batch_size):
            for j in range(self.max_candidate_num):
                if docid_inputs[i][j] < 0:
                    docid_inputs[i][j] = letor_features_length

        ## construct batch_docid_inputs (list of float32 nparrays with shape [query_num])
        ## and batch_labels (list of float32 nparrays with shape [query_num])
        batch_docid_inputs = []
        batch_labels = []
        for length_idx in range(self.max_candidate_num):
            batch_docid_inputs.append(
                np.array(
                    [
                        docid_inputs[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )
            batch_labels.append(
                np.array(
                    [
                        labels[batch_idx][length_idx]
                        for batch_idx in range(local_batch_size)
                    ],
                    dtype=np.float32,
                )
            )

        ## Create input feed map
        input_feed = {}
        input_feed["letor_features"] = np.array(letor_features)
        for l in range(self.max_candidate_num):
            input_feed[f"docid_input{l}"] = batch_docid_inputs[l]
            input_feed[f"label{l}"] = batch_labels[l]

        return input_feed


def create_input_feed(input_feed, list_size, device):
    labels = []
    docid_inputs = []
    letor_features = input_feed["letor_features"]
    for i in range(list_size):
        docid_inputs.append(input_feed[f"docid_input{i}"])
        labels.append(input_feed[f"label{i}"])
    labels = np.transpose(labels)
    labels = torch.from_numpy(labels).to(device)
    docid_inputs = np.array(docid_inputs)
    docid_inputs = torch.as_tensor(data=docid_inputs, dtype=torch.int64)

    return docid_inputs, letor_features, labels
