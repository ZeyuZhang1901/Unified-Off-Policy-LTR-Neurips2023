import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import json

from ranker.AbstractRanker import AbstractRanker
from utils.input_feed import create_input_feed
from network.DQN_CQL import DDQN
from utils import metrics


class DQN_CQLRanker(AbstractRanker):
    def __init__(
        self,
        hyper_json_file,
        feature_size,
        rank_list_size,
        max_visuable_size,
        click_model,
    ):
        with open(hyper_json_file) as ranker_json:
            hypers = json.load(ranker_json)
        # learning rates
        self.policy_lr = hypers["policy_lr"]
        self.embed_lr = hypers["embed_lr"]
        # state type and embedding
        self.state_type = hypers["state_type"]
        self.embed_type = hypers["embed_type"]  # if not using, set None
        self.num_layer = hypers["num_layer"]
        # others
        self.batch_size = hypers["batch_size"]
        self.discount = hypers["discount"]
        self.l2_loss = hypers["l2_loss"]
        self.max_gradient_norm = hypers["max_gradient_norm"]
        self.target_update_steps = hypers["target_update_steps"]
        self.metric_type = hypers["metric_type"]
        self.metric_topn = hypers["metric_topn"]
        self.objective_metric = hypers["objective_metric"]

        ## parameters from outside
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_size = feature_size
        self.rank_list_size = rank_list_size
        self.max_visuable_size = max_visuable_size
        self.click_model = click_model

        ## state setting and embedding
        if self.state_type == "avg":
            self.state_dim = self.feature_size
        elif self.state_type == "avg_pos":
            self.state_dim = 2 * self.feature_size
        elif self.state_type == "avg_rew":
            self.state_dim = self.feature_size + max_visuable_size
        elif self.state_type == "avg_pos_rew":
            self.state_dim = 2 * self.feature_size + max_visuable_size
        # state embedding model
        if self.embed_type == "RNN":
            self.embed_model = nn.RNN(
                self.state_dim, self.state_dim, num_layers=self.num_layer
            ).to(self.device)
        elif self.embed_type == "LSTM":
            self.embed_model = nn.LSTM(
                self.state_dim, self.state_dim, num_layers=self.num_layer
            ).to(self.device)
        # self.embed_optimizer = optim.Adam(
        #     self.embed_model.parameters(), lr=self.embed_lr
        # )

        ## policy and target network
        self.policy_net = DDQN(
            action_dim=self.feature_size, state_dim=self.state_dim
        ).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        if self.embed_type != "None":
            self.optimizer = optim.Adam(
                [
                    {"params": self.policy_net.parameters()},
                    {"params": self.embed_model.parameters(), "lr": self.embed_lr},
                ],
                lr=self.policy_lr,
            )
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.lr_optimizer = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

        ## record and count
        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.q_summary = {}
        self.eval_summary = {}

    def arrange_batch(
        self,
        input_id_list,
    ):
        """Rearrange batch into (s,a,r,s') format. Note that all inputs are in cpu."""

        local_batch_size = input_id_list.shape[1]
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)

        input_feature_list, cum_input_feature_list = [], []
        position_input_list, reward_input_list = [], []
        candidates_list = []
        previous_rewards = torch.zeros(
            local_batch_size, self.max_visuable_size, dtype=torch.float32
        )
        input_feature_list.append(torch.zeros(local_batch_size, self.feature_size))
        for i in range(len(input_id_list)):
            ## actions
            input_feature_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[i], 0))
            )

            ## cumulative states
            if i == 0:
                cum_input_feature_list.append(torch.zeros_like(input_feature_list[0]))
            else:
                cum_input_feature_list.append(
                    torch.stack(input_feature_list[1 : i + 1], dim=0).mean(dim=0)
                )

            ## position embedding
            position_input_list.append(
                torch.repeat_interleave(
                    torch.sin(
                        i
                        / torch.pow(
                            torch.tensor(10000),
                            torch.arange(self.feature_size) * 2 / self.feature_size,
                        )
                    ).reshape(1, -1),
                    local_batch_size,
                    dim=0,
                )
            )

            ## previous rewards
            reward_input_list.append(copy.deepcopy(previous_rewards))
            previous_rewards[:, i] = torch.index_select(
                self.labels.cpu(), dim=1, index=torch.tensor([i])
            ).flatten()
            if i == len(input_id_list)-1:
                reward_input_list.append(copy.deepcopy(previous_rewards))

        ## Candidates
        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[:, i], 0))
            )

        return (
            input_feature_list,
            cum_input_feature_list,
            position_input_list,
            candidates_list,
            reward_input_list,
        )

    def prepare_input_state_and_candidates(
        self,
        input_feature_list,
        cum_input_feature_list,
        position_input_list,
        candidates_list,
        reward_input_list,
    ):

        if self.state_type == "avg":
            states = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
        elif self.state_type == "avg_pos":
            features = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
            positions = torch.stack(position_input_list)
            states = torch.cat([features, positions], dim=-1)
        elif self.state_type == "avg_rew":
            features = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
            rewards = torch.stack(reward_input_list[:-1])
            states = torch.cat([features, rewards], dim=-1)
        elif self.state_type == "avg_pos_rew":
            features = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
            positions = torch.stack(position_input_list)
            rewards = torch.stack(reward_input_list[:-1])
            states = torch.cat([features, positions, rewards], dim=-1)

        states = states.to(torch.float32)
        candidates = torch.cat(candidates_list, dim=0).to(torch.float32)
        return states, candidates

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.policy_net.train()
        self.target_net.train()
        if self.embed_type != "None":
            self.embed_model.train()

        (
            input_feature_list,
            cum_input_feature_list,
            position_input_list,
            candidates_list,
            reward_input_list,
        ) = self.arrange_batch(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )
        states, candidates = self.prepare_input_state_and_candidates(
            input_feature_list,
            cum_input_feature_list,
            position_input_list,
            candidates_list,
            reward_input_list,
        )

        total_mse_loss, total_cql_loss = 0.0, 0.0
        total_policy_q, total_target_q = 0.0, 0.0
        total_norm = 0.0
        ## Update for each position
        local_batch_size = len(candidates_list)
        for i in range(self.max_visuable_size):
            ## prepare for RL elements (states, actions, etc.)
            tmp_states = (
                (
                    states[i, :, :]
                    if self.embed_type == "None"
                    else states[: i + 1, :, :]
                )
                # .detach()
                .to(self.device)
            )
            if self.embed_type != "None":
                tmp_states, _ = self.embed_model(tmp_states)
                tmp_states = tmp_states[-1]

            tmp_actions = (torch.ones(local_batch_size, 1, dtype=torch.float32) * i).to(
                self.device
            )

            tmp_rewards = (
                (reward_input_list[-1][:, i].unsqueeze(-1)).to(self.device)
            )

            if i == self.max_visuable_size - 1:
                tmp_next_states = None
            else:
                tmp_next_states = (
                    (
                        states[i + 1, :, :]
                        if self.embed_type == "None"
                        else states[: i + 2, :, :]
                    )
                    # .detach()
                    .to(self.device)
                )
                if self.embed_type != "None":
                    tmp_next_states, _ = self.embed_model(tmp_next_states)
                    tmp_next_states = tmp_next_states[-1]

            # tmp_candidates = candidates.detach().to(self.device)
            tmp_candidates = candidates.to(self.device)

            ## q values
            q_values = self.policy_net(tmp_states, tmp_candidates)
            q_chosen = q_values.gather(1, tmp_actions.long())
            q_values = q_values[:, i:]
            with torch.no_grad():
                if i == self.max_visuable_size - 1:
                    q_targets = tmp_rewards
                else:
                    q_target_next = self.target_net(tmp_next_states, tmp_candidates)
                    q_targets = (
                        tmp_rewards
                        + self.discount * q_target_next[:, i + 1].unsqueeze(-1).detach()
                    )

            ## loss
            cql_loss = torch.logsumexp(q_values, dim=1).mean() - q_values.mean()
            mse_loss = F.mse_loss(q_chosen, q_targets)
            total_loss = cql_loss + 0.5 * mse_loss

            ## update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_gradient_norm
            )
            self.optimizer.step()
            self.lr_optimizer.step()

            total_mse_loss += mse_loss.detach().item()
            total_cql_loss += cql_loss.detach().item()
            total_policy_q += q_chosen.detach().mean().item()
            total_target_q += q_targets.detach().mean().item()

            ## norm
            critic_norm = 0.0
            for param in self.policy_net.parameters():
                param_norm = param.grad.data.norm(2)
                critic_norm += param_norm.item() ** 2
            critic_norm = critic_norm ** (1.0 / 2)
            total_norm += critic_norm

        if self.global_step % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        ## Loss Summary
        self.loss_summary["loss_mse"] = total_mse_loss / self.max_visuable_size
        self.loss_summary["loss_cql"] = total_cql_loss / self.max_visuable_size
        ## Norm Summary
        self.norm_summary["Gradient Norm"] = total_norm / self.max_visuable_size
        ## Q Summary
        self.q_summary["Policy_Q"] = total_policy_q / self.max_visuable_size
        self.q_summary["Target_Q"] = total_target_q / self.max_visuable_size

        print(
            "Step %d\t%s\t%s"
            % (
                self.global_step,
                "\t".join(
                    [
                        "%s:%.3f" % (key, value)
                        for key, value in self.loss_summary.items()
                    ]
                ),
                "\t".join(
                    ["%s:%.3f" % (key, value) for key, value in self.q_summary.items()]
                ),
            )
        )
        self.global_step += 1
        return self.loss_summary, self.norm_summary, self.q_summary

    def validation(self, input_feed):
        self.policy_net.eval()
        self.target_net.eval()
        if self.embed_type != "None":
            self.embed_model.eval()
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
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)

        input_feature_list = []
        candidates_list = []
        docid_list = []  # store chosen documents

        ## candidates construct
        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(
                    np.take(self.letor_features, self.docid_inputs[:, i], 0)
                )
            )
        candidates = torch.cat(candidates_list, dim=0).to(self.device)

        masks = torch.ones(
            local_batch_size, self.rank_list_size, dtype=torch.float32
        ).to(self.device)
        output = torch.zeros(
            local_batch_size,
            self.rank_list_size,
            dtype=torch.float32,
            device=self.device,
        )
        if self.embed_type != "None":
            h_state = (
                torch.zeros(self.num_layer, local_batch_size, self.state_dim)
                .to(self.device)
                .to(torch.float32)
            )  # hidden state in embedding
            if self.embed_type == "LSTM":
                c_state = (
                    torch.zeros(self.num_layer, local_batch_size, self.state_dim)
                    .to(self.device)
                    .to(torch.float32)
                )  # long-time hidden state in embedding

        ## construct rank list
        for i in range(self.max_visuable_size):
            ## avg features
            if i == 0:
                cum_input_feature = torch.zeros(local_batch_size, self.feature_size)
            else:
                cum_input_feature = sum(input_feature_list) / len(input_feature_list)

            ## positions embedding
            position_input = torch.repeat_interleave(
                torch.sin(
                    i
                    / torch.pow(
                        torch.tensor(10000),
                        torch.arange(self.feature_size) * 2 / self.feature_size,
                    )
                ).reshape(1, -1),
                local_batch_size,
                dim=0,
            )

            ## previous rewards (regard as all-zero)
            rewards = torch.zeros(local_batch_size, self.max_visuable_size)

            ## get index of actions for each query (index: tensor([batch_size]))
            if self.state_type == "avg":
                states = (
                    cum_input_feature
                    if self.embed_type == "None"
                    else (
                        torch.zeros(local_batch_size, self.feature_size)
                        if i == 0
                        else input_feature_list[-1]
                    )
                )
            elif self.state_type == "avg_pos":
                features = (
                    cum_input_feature
                    if self.embed_type == "None"
                    else (
                        torch.zeros(local_batch_size, self.feature_size)
                        if i == 0
                        else input_feature_list[-1]
                    )
                )
                states = torch.cat([features, position_input], dim=1)
            elif self.state_type == "avg_rew":
                features = (
                    cum_input_feature
                    if self.embed_type == "None"
                    else (
                        torch.zeros(local_batch_size, self.feature_size)
                        if i == 0
                        else input_feature_list[-1]
                    )
                )
                states = torch.cat(
                    [features, rewards],
                    dim=1,
                )
            elif self.state_type == "avg_pos_rew":
                features = (
                    cum_input_feature
                    if self.embed_type == "None"
                    else (
                        torch.zeros(local_batch_size, self.feature_size)
                        if i == 0
                        else input_feature_list[-1]
                    )
                )
                states = torch.cat(
                    [features, position_input, rewards],
                    dim=1,
                )

            states = states.to(torch.float32).to(self.device)
            if self.embed_type == "RNN":
                states, h_state = self.embed_model(
                    states.reshape(-1, local_batch_size, self.state_dim), h_state
                )
            elif self.embed_type == "LSTM":
                states, (h_state, c_state) = self.embed_model(
                    states.reshape(-1, local_batch_size, self.state_dim),
                    (h_state, c_state),
                )
            states = states.reshape(1, -1, self.state_dim).squeeze()

            # q_values = self.policy_net(states, candidates)
            # q_values = q_values * masks
            q_values = self.policy_net(states, candidates) * masks
            index = q_values.max(dim=1)[1].flatten()

            docid_list.append(  # list of [1 * batch_size] tensors
                torch.gather(self.docid_inputs, dim=0, index=index.reshape(1, -1))
            )
            input_feature_list.append(  # batch_size * feature_dim
                torch.from_numpy(
                    np.take(self.letor_features, docid_list[i].flatten(), 0)
                )
            )

            output[torch.arange(local_batch_size), index] = self.rank_list_size - i
            masks[torch.arange(local_batch_size), index] = 0.0

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
                torch.ones_like(output_scores[i], device=self.device) * -1e6,
            )
        return torch.stack(output_scores, dim=1)
