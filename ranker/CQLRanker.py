import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random

from ranker.AbstractRanker import AbstractRanker
from ranker.input_feed import create_input_feed
from network.CQL import Actor, Critic
from utils import metrics


class CQLRanker(AbstractRanker):
    def __init__(
        self,
        feature_dim,
        click_model,
        learning_rate,
        rank_list_size,
        metric_type,
        metric_topn,
        state_type,  # (str) what type of state is using
        objective_metric="ndcg_10",
        target_update_step=50,  # target model update every ~ steps
        max_gradient_norm=5.0,  # Clip gradients to this norm.
        batch_size=256,
        discount=0.9,
        max_visuable_size=10,  # max length user can see
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
        self.max_gradient_norm = max_gradient_norm
        self.click_model = click_model
        self.state_type = state_type

        ## soft actor-critic alpha
        # self.log_alpha = torch.tensor([0.0], requires_grad=True)
        # self.alpha = self.log_alpha.exp().detach()
        # self.alpha_optimizer = optim.Adam(
        #     params=[self.log_alpha], lr=self.learning_rate
        # )
        # self.target_entropy = (
        #     -self.max_visuable_size
        # )  # each query with `max_visuable_size` actions to choose

        # self.log_alpha = torch.tensor([-5])  # fix alpha now
        # self.log_alpha =  torch.tensor([-4]) # fix alpha now
        self.log_alpha = torch.tensor([-6])  # fix alpha now
        self.alpha = self.log_alpha.exp()

        ## CQL params
        self.with_lagrange = False  # whether using lagrange
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(
            params=[self.cql_log_alpha], lr=self.learning_rate
        )

        ## Actor network
        self.actor = Actor(self.feature_dim, self.state_type, self.max_visuable_size).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )

        ## Critic network
        self.critic1 = Critic(self.feature_dim, self.state_type, self.max_visuable_size).to(self.device)
        self.critic2 = Critic(self.feature_dim, self.state_type, self.max_visuable_size).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=self.learning_rate
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=self.learning_rate
        )
        self.softmax = nn.Softmax(dim=-1)

        self.metric_type = metric_type
        self.metric_topn = metric_topn
        self.objective_metric = objective_metric

        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.q_summary = {}
        self.eval_summary = {}

    def get_action(self, state, candidates, masks):
        """Return actions for given state as per current policy"""

        with torch.no_grad():
            index = self.actor.get_action(state, candidates, masks)
        return index.cpu()

    def arrange_batch(
        self,
        input_id_list,
    ):
        """Rearrange batch into (s,a,r,s') format."""

        local_batch_size = input_id_list.shape[1]
        PAD_embed = np.zeros((1, self.feature_dim), dtype=np.float32)
        self.letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list, cum_input_feature_list = [], []
        position_input_list, reward_input_list = [], []
        # position_input_list = []
        candidates_list = []
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
                    torch.stack(input_feature_list[:i], dim=0).mean(dim=0)
                )

            ## position states
            # position_input_list.append(  ## one-hot position
            #     F.one_hot(
            #         torch.ones(local_batch_size, dtype=torch.int64) * i,
            #         self.rank_list_size,
            #     ).to(self.device)
            # )
            # position_input_list.append(  ## singal position
            #     (i * torch.ones(local_batch_size, 1)).to(self.device)
            # )
            position_input_list.append(  ## position embedding
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
                )
            )

            ## rewards
            reward_input_list.append(
                torch.index_select(self.labels.cpu(), dim=1, index=torch.tensor([i]))
            )

        ## Candidates
        for i in range(local_batch_size):
            candidates_list.append(
                torch.from_numpy(np.take(self.letor_features, input_id_list[:, i], 0))
            )

        ## All in CPU
        return (
            input_feature_list,
            cum_input_feature_list,
            position_input_list,
            candidates_list,
            reward_input_list,
        )

    def calc_policy_loss(self, states, candidates, masks, alpha):
        """Calculate policy loss"""

        _, action_probs, log_action_probs = self.actor.evaluate(
            states, candidates, masks
        )

        q1 = self.critic1(states, candidates)
        q2 = self.critic2(states, candidates)
        min_q = torch.min(q1, q2)
        actor_loss = (
            (action_probs * (alpha.to(self.device) * log_action_probs - min_q))
            .sum(1)
            .mean()
        )
        # actor_loss = (action_probs * (-min_q)).sum(1).mean()
        log_action_pi = torch.sum(log_action_probs * action_probs, dim=1)

        return actor_loss, log_action_pi

    def update_actor(self, states, candidates, masks, index):
        """Update policy and alpha.\\
        Actor_loss = alpha * log_pi(a | s) - Q(s, a)\\
        actor's target is to map (state, candidates) to action
        """

        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(
            states, candidates, masks, current_alpha
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha_loss = -(
        #     self.log_alpha.exp()
        #     * (log_pis.cpu() + self.target_entropy + index).detach().cpu()
        # ).mean()
        # self.alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()
        # self.alpha = self.log_alpha.exp().detach()

        # return actor_loss.item(), alpha_loss.item()
        return actor_loss.item()

    def cal_target(self, next_states, candidates, rewards, masks, index):
        """Calculate target, get predicted next_state actions and Q-values from target models\\
        Then compute critic loss.

        Q_target = r + discount * min(critic1(next_state, actor(next_state)), critic2(next_state, actor(next_state)))
        -alpha * log_pi(next_action | next_state)\\
        Critic_loss = MSELoss(Q, Q_target)

        Critic's target is to map (state, action) to Q values
        """

        if next_states is None:  ## no next-state exist, use reward as targets
            return rewards.to(self.device)

        else:  ## calculate Q target with next_state
            mask_current = copy.deepcopy(masks)
            mask_current[:, index] = 0.0  # mask current position
            with torch.no_grad():
                _, action_probs, log_pis = self.actor.evaluate(
                    next_states, candidates, mask_current
                )
                q1_target = self.critic1_target(next_states, candidates)
                q2_target = self.critic2_target(next_states, candidates)
                q_target = action_probs * (
                    torch.min(q1_target, q2_target)
                    - self.alpha.to(self.device) * log_pis
                )
                # q_target = action_probs * torch.min(q1_target, q2_target)

                ## Compute Q targets for current states
                q_targets = rewards.to(self.device) + self.discount * q_target.sum(
                    dim=1
                ).unsqueeze(-1)

                return q_targets

    def update_critic(
        self, states, actions, rewards, next_states, candidates, masks, index
    ):
        """update critic. note that actions here are implemented as numbers (int)
        The last parameter indicate current position"""

        q_targets = self.cal_target(next_states, candidates, rewards, masks, index)

        ## compute critic loss
        q1 = self.critic1(states, candidates)
        q2 = self.critic2(states, candidates)
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())

        q1 = q1[:, index:]
        q2 = q2[:, index:]

        critic1_loss = 0.5 * F.mse_loss(q1_, q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, q_targets)

        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])

        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1e6).to(
                self.device
            )
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        ## Update critics
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_gradient_norm)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_gradient_norm)
        self.critic2_optimizer.step()

        return (
            critic1_loss.item(),
            critic2_loss.item(),
            cql1_scaled_loss.item(),
            cql2_scaled_loss.item(),
            cql_alpha_loss.item(),
            q1_.mean().item(),
            q2_.mean().item(),
            q_targets.mean().item(),
        )

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic1_target.train()
        self.critic2_target.train()

        (
            input_feature_list,  # action
            cum_input_feature_list,
            position_input_list,  # position (state)
            candidates_list,  # candidates
            reward_input_list,  # rewards
        ) = self.arrange_batch(
            input_id_list=self.docid_inputs[: self.max_visuable_size]
        )

        # total_actor_loss, total_alpha_loss = 0, 0
        total_actor_loss = 0
        total_critic1_loss, total_critic2_loss = 0, 0
        total_cql1_loss, total_cql2_loss, total_cql_alpha_loss = 0, 0, 0
        total_q1_values, total_q2_values, total_q_target_values = 0, 0, 0

        ## Update for each position
        candidates = torch.cat(candidates_list, dim=0)
        local_batch_size = len(candidates_list)
        masks = torch.ones(
            local_batch_size, self.max_visuable_size, dtype=torch.float32
        )
        for i in range(self.max_visuable_size):
            ## arrange input as (s, a, r, s')

            if self.state_type == "pos":
                states = position_input_list[i]
            elif self.state_type == "avg":
                states = cum_input_feature_list[i]
            elif self.state_type == "pos_avg":
                states = torch.cat(
                    [cum_input_feature_list[i], position_input_list[i]], dim=1
                )
            elif self.state_type == "pos_avg_rew":
                rewards_all = torch.cat(reward_input_list, dim=1)
                rewards_all[:, i:] = 0
                states = torch.cat(
                    [cum_input_feature_list[i], position_input_list[i], rewards_all],
                    dim=1,
                )

            actions = (torch.ones(local_batch_size, 1) * i).to(self.device)

            rewards = reward_input_list[i]

            if self.state_type == "pos":
                next_states = (
                    None
                    if i == self.max_visuable_size - 1
                    else position_input_list[i + 1]
                )
            elif self.state_type == "avg":
                next_states = (
                    None
                    if i == self.max_visuable_size - 1
                    else cum_input_feature_list[i + 1]
                )
            elif self.state_type == "pos_avg":
                next_states = (
                    None
                    if i == self.max_visuable_size - 1
                    else torch.cat(
                        [cum_input_feature_list[i + 1], position_input_list[i + 1]],
                        dim=1,
                    )
                )
            elif self.state_type == "pos_avg_rew":
                rewards_all = torch.cat(reward_input_list, dim=1)
                if i == self.max_visuable_size - 1:
                    rewards_all = 0
                else:
                    rewards_all[:, i + 1 :] = 0
                next_states = (
                    None
                    if i == self.max_visuable_size - 1
                    else torch.cat(
                        [
                            cum_input_feature_list[i + 1],
                            position_input_list[i + 1],
                            rewards_all,
                        ],
                        dim=1,
                    )
                )

            ## update actor
            # actor_loss, alpha_loss = self.update_actor(
            #     position_input_list[i], candidates, masks, i
            # )
            actor_loss = self.update_actor(
                states,
                candidates,
                masks,
                i,
            )
            total_actor_loss += actor_loss
            # total_alpha_loss += alpha_loss

            ## update critics
            (
                critic1_loss,
                critic2_loss,
                cql1_scaled_loss,
                cql2_scaled_loss,
                cql_alpha_loss,
                q1_values,
                q2_values,
                q_target_values,
            ) = self.update_critic(
                states,
                actions,
                rewards,
                next_states,
                candidates,
                masks,
                i,  # indicate current position
            )
            total_critic1_loss += critic1_loss
            total_critic2_loss += critic2_loss
            total_cql1_loss += cql1_scaled_loss
            total_cql2_loss += cql2_scaled_loss
            total_cql_alpha_loss += cql_alpha_loss
            total_q1_values += q1_values
            total_q2_values += q2_values
            total_q_target_values += q_target_values

            ## Change masks
            masks[:, i] = 0.0

        ## Calculate norm
        total_norm = 0
        actor_params = self.actor.parameters()
        critic1_params = self.critic1.parameters()
        critic2_params = self.critic2.parameters()
        model_params = list(actor_params) + list(critic1_params) + list(critic2_params)
        for param in model_params:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.norm = total_norm

        ## Update target network
        if self.global_step % self.target_update_step == 0:
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())

        ## Loss Summary
        self.loss_summary["Actor"] = total_actor_loss / self.max_visuable_size
        # self.loss_summary["Alpha"] = total_alpha_loss / self.max_visuable_size
        self.loss_summary["Critic1"] = total_critic1_loss / self.max_visuable_size
        self.loss_summary["Critic2"] = total_critic2_loss / self.max_visuable_size
        self.loss_summary["Cql1"] = total_cql1_loss / self.max_visuable_size
        self.loss_summary["Cql2"] = total_cql2_loss / self.max_visuable_size
        self.loss_summary["Cql Alpha"] = total_cql_alpha_loss / self.max_visuable_size
        ## Norm Summary
        self.norm_summary["Gradient Norm"] = self.norm
        ## Q Summary
        self.q_summary["Critic_Q1"] = total_q1_values / self.max_visuable_size
        self.q_summary["Critic_Q2"] = total_q2_values / self.max_visuable_size
        self.q_summary["Target_Q"] = total_q_target_values / self.max_visuable_size

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

    # %%
    # TODO Validation step and its auxiliary functions
    def validation(self, input_feed):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
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
        candidates = torch.cat(candidates_list, dim=0)

        masks = torch.ones(local_batch_size, self.rank_list_size, dtype=torch.float32)
        output = torch.zeros(
            local_batch_size,
            self.rank_list_size,
            dtype=torch.float32,
            device=self.device,
        )
        ## construct rank list for each sampled query
        # for i in range(self.rank_list_size):  # for each rank position
        for i in range(self.max_visuable_size):  # for each considered rank position
            ## avg features state
            if i == 0:
                cum_input_feature = torch.zeros(local_batch_size, self.feature_dim)
            else:
                cum_input_feature = sum(input_feature_list) / len(input_feature_list)

            ## positions state
            # # one-hot position
            # position_input = F.one_hot(
            #     torch.ones(local_batch_size, dtype=torch.int64) * i,
            #     self.rank_list_size,
            # ).to(self.device)
            # # single position
            # position_input = (i * torch.ones(local_batch_size, 1)).to(self.device)
            # position-embedding
            position_input = torch.repeat_interleave(
                torch.sin(
                    i
                    / torch.pow(
                        torch.tensor(10000),
                        torch.arange(self.feature_dim) * 2 / self.feature_dim,
                    )
                ).reshape(1, -1),
                local_batch_size,
                dim=0,
            )

            ## get index of actions for each query (index: tensor([batch_size]))
            if self.state_type == "pos":
                index = self.get_action(position_input, candidates, masks)
            elif self.state_type == "avg":
                index = self.get_action(cum_input_feature, candidates, masks)
            elif self.state_type == "pos_avg":
                index = self.get_action(
                    torch.cat([cum_input_feature, position_input], dim=1),
                    candidates,
                    masks,
                )
            elif self.state_type == "pos_avg_rew":
                index = self.get_action(
                    torch.cat(
                        [
                            cum_input_feature,
                            position_input,
                            torch.zeros(local_batch_size, self.max_visuable_size),
                        ],
                        dim=1,
                    ),
                    candidates,
                    masks,
                )

            docid_list.append(  # list of [1 * batch_size] tensors
                torch.gather(self.docid_inputs, dim=0, index=index.cpu().reshape(1, -1))
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
