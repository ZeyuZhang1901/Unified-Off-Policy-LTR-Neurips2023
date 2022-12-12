import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from network.CQL import Actor, Critic, Scalar
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "chosen", "qid")
)


class CQLRanker(AbstractRanker):
    def __init__(
        self,
        state_dim,
        action_dim,
        batch_size,
        train_set,
        test_set,
        lr=1e-3,
        discount=0.9,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # actor part: get action and probability distribution
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # critic part: get scores for each state-action pair
        self.qf1 = Critic(state_dim, action_dim)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.qf2 = Critic(state_dim, action_dim)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.critic_optimizer = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr
        )
        # alpha tuning
        self.log_alpha = Scalar(0.0)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=lr)
        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(), lr=lr
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount = discount
        self.train_set = train_set
        self.test_set = test_set

    def update_policy(self, memory):
        # get batch
        trainsitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*trainsitions))
        states = torch.cat(batch.state).to(self.device)
        # state_batch = torch.zeros_like(torch.cat(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        nextstates = torch.cat(batch.next_state).to(self.device)
        # next_state_batch = torch.zeros_like(torch.cat(batch.next_state), dtype=torch.float32).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        dones = torch.cat(batch.done).to(self.device)

        # actor get new action and prob for each state in batch
        new_actions = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        new_indexs = torch.zeros(self.batch_size, dtype=torch.int).to(self.device)
        new_probs = torch.zeros(self.batch_size, dtype=torch.float32).to(self.device)
        probs = []
        for i in range(self.batch_size):
            candidates = torch.tensor(
                self.train_set.get_all_features_by_query(batch.qid[i])
            ).to(self.device)
            chosen = torch.tensor(batch.chosen[i]).to(self.device)
            new_indexs[i], new_actions[i], prob = self.actor(
                states[i], candidates, chosen
            )
            new_probs[i] = prob[new_indexs[i]]
            probs.append(prob)

        # tuning alpha (automatic entropy tuning)
        alpha_loss = -(self.log_alpha() * torch.log(new_probs).detach()).mean()
        alpha = self.log_alpha().exp()

        # actor loss
        q_new_actions = torch.min(
            self.qf1(states, new_actions),
            self.qf2(states, new_actions),
        )
        actor_loss = (alpha * torch.log(new_probs) - q_new_actions).mean()

        # critic loss (q functions)
        q1_pred = self.qf1(states, actions)
        q2_pred = self.qf2(states, actions)

        new_next_actions = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        new_next_indexs = torch.zeros(self.batch_size, dtype=torch.int).to(self.device)
        new_next_probs = torch.zeros(self.batch_size, dtype=torch.float32).to(
            self.device
        )
        next_probs = []
        for i in range(self.batch_size):
            candidates = torch.tensor(
                self.train_set.get_all_features_by_query(batch.qid[i])
            ).to(self.device)
            chosen = torch.tensor(batch.chosen[i]).to(self.device)
            index = self.train_set.get_docid_by_query_and_feature(
                batch.qid[i], actions[i]
            )
            chosen[index] = False
            new_next_indexs[i], new_next_actions[i], next_prob = self.actor(
                nextstates[i], candidates, chosen
            )
            new_next_probs[i] = prob[new_indexs[i]]
            next_probs.append(next_prob)
        target_q_values = torch.min(
            self.target_qf1(nextstates, new_next_actions),
            self.target_qf2(nextstates, new_next_actions),
        )
        td_target = rewards + (1 - dones) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred, td_target.detach())
        qf2_loss = F.mse_loss(q2_pred, td_target.detach())

        # add CQL term to qf loss
