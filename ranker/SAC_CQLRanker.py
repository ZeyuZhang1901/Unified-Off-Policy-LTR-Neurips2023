import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import json

from ranker.AbstractRanker import AbstractRanker
from utils.input_feed import create_input_feed
from network.SAC_CQL import Actor, Critic
from utils import metrics


class SAC_CQLRanker(AbstractRanker):
    def __init__(
        self,
        hyper_json_file,
        feature_size,
        rank_list_size,
        max_visuable_size,  # max length user can see
        click_model,
    ):
        ## load hyperparameters from json file
        with open(hyper_json_file) as ranker_json:
            hypers = json.load(ranker_json)
        # learning rates
        self.actor_lr = hypers["actor_lr"]
        self.critic_lr = hypers["critic_lr"]
        self.alpha_lr = hypers["alpha_lr"]
        self.cql_alpha_lr = hypers["cql_alpha_lr"]
        self.embed_lr = hypers["embed_lr"]
        self.lr_decay_type = hypers["lr_decay_type"]  # if no decay, set "None"
        # state type and embedding
        self.state_type = hypers["state_type"]
        self.embed_type = hypers["embed_type"]  # if not using, set "None"
        self.num_layer = hypers["num_layer"]
        self.update_embed = eval(hypers["update_embed"])
        # actor
        self.auto_actor_alpha = eval(hypers["auto_actor_alpha"])
        self.initial_log_alpha = hypers["initial_log_alpha"]
        self.actor_node_list = hypers["actor_structure"]
        # critic and cql
        self.critic_node_list = hypers["critic_structure"]
        self.using_cql = eval(hypers["using_cql"])
        self.with_lagrange = eval(hypers["with_lagrange"])
        self.target_action_gap = hypers["target_action_gap"]
        self.initial_log_cql_alpha = hypers["initial_log_cql_alpha"]
        # others
        self.ac_update_step = hypers["ac_update_step"]  # if 0, update together.
        self.embed_update_step = hypers["embed_update_step"]  # if 0, update together.
        self.batch_size = hypers["batch_size"]
        self.discount = hypers["discount"]
        self.l2_loss = hypers["l2_loss"]
        self.max_gradient_norm = hypers["max_gradient_norm"]
        self.soft_update = hypers["soft_update"]
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
        # elif self.state_type == "avg_pos":
        #     self.state_dim = 2 * self.feature_size
        elif self.state_type == "avg_rew":
            self.state_dim = self.feature_size + max_visuable_size

        # state embedding model
        if self.embed_type == "RNN":
            self.embed_model = nn.RNN(
                self.state_dim, self.state_dim, num_layers=self.num_layer
            ).to(self.device)
            self.embed_optimizer = optim.Adam(
                self.embed_model.parameters(), lr=self.embed_lr
            )
            if self.lr_decay_type == "exp":
                self.embed_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                    self.embed_optimizer, gamma=0.9993
                )
            elif self.lr_decay_type == "step":
                self.embed_lr_optimizer = optim.lr_scheduler.StepLR(
                    self.embed_optimizer, step_size=5000, gamma=0.1
                )
        elif self.embed_type == "LSTM":
            self.embed_model = nn.LSTM(
                self.state_dim, self.state_dim, num_layers=self.num_layer
            ).to(self.device)
            self.embed_optimizer = optim.Adam(
                self.embed_model.parameters(), lr=self.embed_lr
            )
            if self.lr_decay_type == "exp":
                self.embed_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                    self.embed_optimizer, gamma=0.9993
                )
            elif self.lr_decay_type == "step":
                self.embed_lr_optimizer = optim.lr_scheduler.StepLR(
                    self.embed_optimizer, step_size=5000, gamma=0.1
                )

        ## Actor net
        self.actor = Actor(
            action_dim=self.feature_size,
            state_dim=self.state_dim + self.feature_size,  # add dimension for action
            num_node_list=self.actor_node_list,
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        if self.lr_decay_type == "step":
            self.actor_lr_optimizer = optim.lr_scheduler.StepLR(
                self.actor_optimizer, step_size=5000, gamma=0.1
            )
        elif self.lr_decay_type == "exp":
            self.actor_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                self.actor_optimizer, gamma=0.9993
            )

        # alpha
        if self.auto_actor_alpha:  # learn actor alpha automatically
            self.log_alpha = torch.tensor([self.initial_log_alpha], requires_grad=True)
            self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)
            if self.lr_decay_type == "exp":
                self.alpha_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                    self.alpha_optimizer, gamma=0.9993
                )
            elif self.lr_decay_type == "step":
                self.alpha_lr_optimizer = optim.lr_scheduler.StepLR(
                    self.alpha_optimizer, step_size=5000, gamma=0.1
                )
            self.target_entropy = (
                -self.max_visuable_size
            )  # each query with `max_visuable_size` actions to choose
        else:
            self.log_alpha = torch.tensor([self.initial_log_alpha])  # fix alpha now
        self.alpha = self.log_alpha.exp().detach()

        ## Critic net
        self.critic1 = Critic(
            action_dim=self.feature_size,
            state_dim=self.state_dim + self.feature_size,  # add dimension for action
            num_node_list=self.critic_node_list,
        ).to(self.device)
        self.critic2 = Critic(
            action_dim=self.feature_size,
            state_dim=self.state_dim + self.feature_size,  # add dimension for action
            num_node_list=self.critic_node_list,
        ).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.critic_lr,
        )
        if self.lr_decay_type == "step":
            self.critic_lr_optimizer = optim.lr_scheduler.StepLR(
                self.critic_optimizer, step_size=5000, gamma=0.1
            )
        elif self.lr_decay_type == "exp":
            self.critic_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                self.critic_optimizer, gamma=0.9993
            )

        # cql 
        if self.using_cql:
            self.cql_log_alpha = torch.tensor(
                [self.initial_log_cql_alpha], requires_grad=True
            )
            self.cql_alpha_optimizer = optim.Adam(
                params=[self.cql_log_alpha], lr=self.cql_alpha_lr
            )
            if self.lr_decay_type == "step":
                self.cql_alpha_lr_optimizer = optim.lr_scheduler.StepLR(
                    self.cql_alpha_optimizer, step_size=5000, gamma=0.1
                )
            elif self.lr_decay_type == "exp":
                self.cql_alpha_lr_optimizer = optim.lr_scheduler.ExponentialLR(
                    self.cql_alpha_optimizer, gamma=0.9993
                )

        ## record and count
        if self.ac_update_step > 0 and self.embed_update_step > 0:
            self.update_flag = 0  # 0: update ac, 1: update embed
            self.update_count = 0
        self.global_step = 0
        self.loss_summary = {}
        self.norm_summary = {}
        self.q_summary = {}
        self.eval_summary = {}

    def get_action(
        self,
        state,
        candidates,
        masks,
    ):
        """Return actions for given state as per current policy"""

        with torch.no_grad():
            index = self.actor.get_action(state, candidates, masks)
        return index.cpu()

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
            if i == len(input_id_list) - 1:
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
        candidates_list,
        reward_input_list,
    ):

        if self.state_type == "avg":
            states = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
        elif self.state_type == "avg_rew":
            features = (
                torch.stack(input_feature_list)[:-1]
                if self.embed_type != "None"
                else torch.stack(cum_input_feature_list)
            )
            rewards = torch.stack(reward_input_list[:-1])
            states = torch.cat([features, rewards], dim=-1)

        states = states.to(torch.float32)
        candidates = torch.cat(candidates_list, dim=0).to(torch.float32)
        return states, candidates

    def update_policy(self, input_feed):
        self.docid_inputs, self.letor_features, self.labels = create_input_feed(
            input_feed, self.max_visuable_size, self.device
        )
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic1_target.train()
        self.critic2_target.train()
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
            candidates_list,
            reward_input_list,
        )
        total_actor_loss, total_alpha_loss = 0, 0
        total_critic1_loss, total_critic2_loss = 0, 0
        total_all_c1_loss, total_all_c2_loss = 0, 0
        total_cql1_loss, total_cql2_loss, total_cql_alpha_loss = 0, 0, 0
        total_q1_values, total_q2_values, total_q_target_values = 0, 0, 0

        ## Update for each position
        local_batch_size = len(candidates_list)
        masks = torch.ones(
            local_batch_size, self.max_visuable_size, dtype=torch.float32
        )
        total_norm = 0.0
        total_actor_norm, total_critic_norm, total_embed_norm = 0, 0, 0
        for i in range(self.max_visuable_size):
            ## update actor
            actor_loss, alpha_loss, actor_norm = self.update_actor(
                states=states[i, :, :]
                if self.embed_type == "None"
                else states[: i + 1, :, :],
                positions = position_input_list[i].to(torch.float32),
                candidates=candidates,
                masks=masks,
                index=i,
            )
            total_actor_loss += actor_loss
            total_alpha_loss += alpha_loss

            ## update critic
            (
                critic1_loss,
                critic2_loss,
                cql1_scaled_loss,
                cql2_scaled_loss,
                cql_alpha_loss,
                q1_values,
                q2_values,
                q_target_values,
                critic_norm,
                embed_norm,
            ) = self.update_critic(
                states=states[i, :, :]
                if self.embed_type == "None"
                else states[: i + 1, :, :],
                actions=torch.ones(local_batch_size, 1, dtype=torch.float32) * i,
                rewards=reward_input_list[-1][:, i].unsqueeze(-1),
                next_states=None
                if i == self.max_visuable_size - 1
                else (
                    states[i + 1, :, :]
                    if self.embed_type == "None"
                    else states[: i + 2, :, :]
                ),
                positions=position_input_list[i].to(torch.float32),
                candidates=candidates,
                masks=masks,
                index=i,
            )
            total_critic1_loss += critic1_loss.detach().item()
            total_critic2_loss += critic2_loss.detach().item()
            total_cql1_loss += cql1_scaled_loss.detach().item()
            total_cql2_loss += cql2_scaled_loss.detach().item()
            total_all_c1_loss += (
                critic1_loss.detach().item() + cql1_scaled_loss.detach().item()
            )
            total_all_c2_loss += (
                critic2_loss.detach().item() + cql2_scaled_loss.detach().item()
            )
            total_cql_alpha_loss += cql_alpha_loss.detach().item()
            total_q1_values += q1_values.detach().item()
            total_q2_values += q2_values.detach().item()
            total_q_target_values += q_target_values.detach().item()

            ## Change masks
            masks[:, i] = 0.0

            ## add norm
            total_actor_norm += actor_norm
            total_critic_norm += critic_norm
            total_embed_norm += embed_norm
            total_norm += actor_norm + critic_norm + embed_norm

        ## Loss Summary
        self.loss_summary["l_actor"] = total_actor_loss / self.max_visuable_size
        self.loss_summary["l_Alpha"] = total_alpha_loss / self.max_visuable_size
        self.loss_summary["l_critic1"] = total_critic1_loss / self.max_visuable_size
        self.loss_summary["l_critic2"] = total_critic2_loss / self.max_visuable_size
        self.loss_summary["l_cql1"] = total_cql1_loss / self.max_visuable_size
        self.loss_summary["l_cql2"] = total_cql2_loss / self.max_visuable_size
        self.loss_summary["l_cql_alpha"] = total_cql_alpha_loss / self.max_visuable_size
        ## Norm Summary
        self.norm_summary["Actor Norm"] = total_actor_norm / self.max_visuable_size
        self.norm_summary["Critic Norm"] = total_critic_norm / self.max_visuable_size
        self.norm_summary["Embed Norm"] = total_embed_norm / self.max_visuable_size
        self.norm_summary["Gradient Norm"] = total_norm / self.max_visuable_size
        ## Q Summary
        self.q_summary["Critic_Q1"] = total_q1_values / self.max_visuable_size
        self.q_summary["Critic_Q2"] = total_q2_values / self.max_visuable_size
        self.q_summary["Target_Q"] = total_q_target_values / self.max_visuable_size

        print(
            "Step %d\t%s\t%s\t%s"
            % (
                self.global_step,
                "\t".join(
                    ["%s:%.3f" % (key, value) for key, value in self.q_summary.items()]
                ),
                "\t".join(
                    [
                        "%s:%.3f" % (key, value)
                        for key, value in self.loss_summary.items()
                    ]
                ),
                "\t".join(
                    [
                        "%s:%.3f" % (key, value)
                        for key, value in self.norm_summary.items()
                    ]
                ),
            )
        )

        self.global_step += 1
        if self.ac_update_step > 0 and self.embed_update_step > 0:
            self.update_count += 1
            if self.update_flag == 0:  # update ac
                if self.update_count % self.ac_update_step == 0:
                    self.update_count = 0
                    self.update_flag = 1
                    print(f"Step {self.global_step}, switch to update embedding model!")
            else:
                if self.update_count % self.embed_update_step == 0:
                    self.update_count = 0
                    self.update_flag = 0
                    print(f"Step {self.global_step}, switch to update actor critirs!")

        return self.loss_summary, self.norm_summary, self.q_summary

    def update_actor(
        self,
        states,
        positions,
        candidates,
        masks,
        index,
    ):
        ## detach from outside, in case error in calculate graph
        tmp_states = states.detach().to(self.device)
        tmp_candidates = candidates.detach().to(self.device)
        tmp_masks = masks.detach().to(self.device)
        tmp_positions = positions.detach().to(self.device)
        # print(tmp_states.dtype)
        if self.embed_type != "None":
            tmp_states, _ = self.embed_model(tmp_states)
            tmp_states = tmp_states[-1]
        tmp_states = torch.cat([tmp_states, tmp_positions], dim=1)

        ## actor loss
        _, action_probs, log_action_probs = self.actor.evaluate(
            tmp_states, tmp_candidates, tmp_masks
        )
        q1 = self.critic1(tmp_states, tmp_candidates)
        q2 = self.critic2(tmp_states, tmp_candidates)
        min_q = torch.min(q1, q2)
        current_alpha = copy.deepcopy(self.alpha).to(self.device)
        actor_loss = (
            (action_probs * (current_alpha * log_action_probs - min_q)).sum(1).mean()
        )

        ## alpha loss
        if not self.auto_actor_alpha:
            alpha_loss = 0
        else:
            log_action_pi = torch.sum(log_action_probs * action_probs, dim=1)
            alpha_loss = -(
                self.log_alpha.exp()
                * (log_action_pi.cpu() + self.target_entropy + index).detach().cpu()
            ).mean()

        ## updates networks
        if self.ac_update_step > 0 and self.embed_update_step > 0:
            if self.update_flag:  # update embed
                self.embed_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.embed_model.parameters(), self.max_gradient_norm
                )
                self.embed_optimizer.step()
                if self.lr_decay_type != "None":
                    self.embed_lr_optimizer.step()
            else:  # update ac
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_gradient_norm
                )
                self.actor_optimizer.step()
                if self.lr_decay_type != "None":
                    self.actor_lr_optimizer.step()
                if self.auto_actor_alpha:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    if self.lr_decay_type != "None":
                        self.alpha_lr_optimizer.step()
        else:  # update actor, embed and actor alpha
            self.actor_optimizer.zero_grad()
            if self.embed_type != "None" and self.update_embed:
                self.embed_optimizer.zero_grad()
            actor_loss.backward()
            if self.embed_type != "None" and self.update_embed:
                torch.nn.utils.clip_grad_norm_(
                    self.embed_model.parameters(), self.max_gradient_norm
                )
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_gradient_norm
            )
            self.actor_optimizer.step()
            if self.embed_type != "None" and self.update_embed:
                self.embed_optimizer.step()
            if self.lr_decay_type != "None":
                self.actor_lr_optimizer.step()
                if self.embed_type != "None" and self.update_embed:
                    self.embed_lr_optimizer.step()
            if self.auto_actor_alpha:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                if self.lr_decay_type != "None":
                    self.alpha_lr_optimizer.step()

        ## actor norm
        actor_norm = 0
        for param in self.actor.parameters():
            param_norm = param.grad.data.norm(2)
            actor_norm += param_norm.item() ** 2
        actor_norm = actor_norm ** (1.0 / 2)

        return actor_loss, alpha_loss, actor_norm

    def update_critic(
        self,
        states,
        actions,
        rewards,
        next_states,
        positions,
        candidates,
        masks,
        index,
    ):
        ## detach from outside, in case error in calculate graph
        tmp_positions = positions.detach().to(self.device)
        tmp_states = states.detach().to(self.device)
        if self.embed_type != "None":
            tmp_states, _ = self.embed_model(tmp_states)
            tmp_states = tmp_states[-1]
        tmp_states = torch.cat([tmp_states, tmp_positions], dim=-1)
        tmp_actions = actions.detach().to(self.device)
        tmp_rewards = rewards.detach().to(self.device)
        if next_states != None:
            tmp_next_states = next_states.detach().to(self.device)
            if self.embed_type != "None":
                tmp_next_states, _ = self.embed_model(tmp_next_states)
                tmp_next_states = tmp_next_states[-1]
            tmp_next_states = torch.cat([tmp_next_states, tmp_positions], dim=-1)
        else:
            tmp_next_states = None
        tmp_candidates = candidates.detach().to(self.device)
        tmp_masks = masks.detach().to(self.device)

        ## calculate targets
        if tmp_next_states is None:
            q_targets = tmp_rewards
        else:
            mask_current = copy.deepcopy(tmp_masks)
            mask_current[:, index] = 0.0  # mask current position
            with torch.no_grad():
                _, action_probs, log_pis = self.actor.evaluate(
                    tmp_next_states, tmp_candidates, mask_current
                )
                q1_target = self.critic1_target(tmp_next_states, tmp_candidates)
                q2_target = self.critic2_target(tmp_next_states, tmp_candidates)
                q_target = action_probs * (
                    torch.min(q1_target, q2_target)
                    - self.alpha.to(self.device) * log_pis
                )
                q_targets = tmp_rewards + self.discount * q_target.sum(1).unsqueeze(-1)

        ## compute critic loss
        q1 = self.critic1(tmp_states, tmp_candidates)
        q2 = self.critic2(tmp_states, tmp_candidates)
        q1_ = q1.gather(1, tmp_actions.long())
        q2_ = q2.gather(1, tmp_actions.long())

        q1 = q1[:, index:]
        q2 = q2[:, index:]

        critic1_loss = 0.5 * F.mse_loss(q1_, q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, q_targets)

        if self.using_cql:
            cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
            cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

            cql_alpha_loss = torch.FloatTensor([0.0])

            if self.with_lagrange:
                cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1e6).to(
                    self.device
                )
                cql1_scaled_loss = cql_alpha * (
                    cql1_scaled_loss - self.target_action_gap
                )
                cql2_scaled_loss = cql_alpha * (
                    cql2_scaled_loss - self.target_action_gap
                )

                cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5

                total_c1_loss = critic1_loss + cql1_scaled_loss
                total_c2_loss = critic2_loss + cql2_scaled_loss
            else:
                cql_log_alpha = torch.FloatTensor([self.initial_log_cql_alpha]).to(
                    self.device
                )
                cql_alpha = torch.exp(cql_log_alpha)
                total_c1_loss = critic1_loss + cql_alpha * cql1_scaled_loss
                total_c2_loss = critic2_loss + cql_alpha * cql2_scaled_loss
        else:
            cql1_scaled_loss = torch.FloatTensor([0.0])
            cql2_scaled_loss = torch.FloatTensor([0.0])
            cql_alpha_loss = torch.FloatTensor([0.0])
            total_c1_loss = critic1_loss
            total_c2_loss = critic2_loss

        total_loss = total_c1_loss + total_c2_loss

        ## update critics
        if self.ac_update_step > 0 and self.embed_update_step > 0:
            if self.update_flag:  # update embed
                self.embed_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.embed_model.parameters(), self.max_gradient_norm
                )
                self.embed_optimizer.step()
                if self.lr_decay_type != "None":
                    self.embed_lr_optimizer.step()
            else:
                if self.using_cql and self.with_lagrange:
                    self.cql_alpha_optimizer.zero_grad()
                    cql_alpha_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.cql_log_alpha,
                        self.max_gradient_norm,
                    )
                    self.cql_alpha_optimizer.step()
                    if self.lr_decay_type != "None":
                        self.cql_alpha_lr_optimizer.step()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.critic1.parameters()) + list(self.critic2.parameters()),
                    self.max_gradient_norm,
                )
                self.critic_optimizer.step()
                if self.lr_decay_type != "None":
                    self.critic_lr_optimizer.step()
        else:  # update critic and embed together
            if self.using_cql and self.with_lagrange:
                self.cql_alpha_optimizer.zero_grad()
                cql_alpha_loss.backward(retain_graph=True)
                self.cql_alpha_optimizer.step()
                if self.lr_decay_type != "None":
                    self.cql_alpha_lr_optimizer.step()
            self.critic_optimizer.zero_grad()
            if self.embed_type != "None" and self.update_embed:
                self.embed_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic1.parameters()) + list(self.critic2.parameters()),
                self.max_gradient_norm,
            )
            if self.embed_type != "None" and self.update_embed:
                torch.nn.utils.clip_grad_norm_(
                    self.embed_model.parameters(), self.max_gradient_norm
                )
            self.critic_optimizer.step()
            if self.embed_type != "None" and self.update_embed:
                self.embed_optimizer.step()
            if self.lr_decay_type != "None":
                self.critic_lr_optimizer.step()
                if self.embed_type != "None" and self.update_embed:
                    self.embed_lr_optimizer.step()

        ## soft update target
        for target_param, local_param in zip(
            self.critic1_target.parameters(), self.critic1.parameters()
        ):
            target_param.data.copy_(
                self.soft_update * local_param.data
                + (1.0 - self.soft_update) * target_param.data
            )
        for target_param, local_param in zip(
            self.critic2_target.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                self.soft_update * local_param.data
                + (1.0 - self.soft_update) * target_param.data
            )

        ## critic norm
        critic_norm = 0.0
        critic1_params = self.critic1.parameters()
        critic2_params = self.critic2.parameters()
        model_params = list(critic1_params) + list(critic2_params)
        for param in model_params:
            param_norm = param.grad.data.norm(2)
            critic_norm += param_norm.item() ** 2
        critic_norm = critic_norm ** (1.0 / 2)

        ## embed norm
        embed_norm = 0.0
        if self.embed_type != "None" and self.update_embed:
            for param in self.embed_model.parameters():
                param_norm = param.grad.data.norm(2)
                embed_norm += param_norm.item() ** 2
            embed_norm = embed_norm ** (1.0 / 2)

        return (
            critic1_loss,
            critic2_loss,
            cql1_scaled_loss,
            cql2_scaled_loss,
            cql_alpha_loss,
            q1_.mean(),
            q2_.mean(),
            q_targets.mean(),
            critic_norm,
            embed_norm,
        )

    def validation(self, input_feed):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
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

        masks = torch.ones(local_batch_size, self.rank_list_size, dtype=torch.float32)
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
            ).to(torch.float32).to(self.device)

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
            # elif self.state_type == "avg_pos":
            #     features = (
            #         cum_input_feature
            #         if self.embed_type == "None"
            #         else (
            #             torch.zeros(local_batch_size, self.feature_size)
            #             if i == 0
            #             else input_feature_list[-1]
            #         )
            #     )
            #     states = torch.cat([features, position_input], dim=1)
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
            states = torch.cat([states, position_input], dim=-1)
            index = self.get_action(states, candidates, masks)

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
