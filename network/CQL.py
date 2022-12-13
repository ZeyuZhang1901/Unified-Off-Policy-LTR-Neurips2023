import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
    ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 250)
        self.ln2 = nn.Linear(250, 250)
        self.ln3 = nn.Linear(250, 1)

    def forward(self, state, action):
        q1 = F.relu(self.ln1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.ln2(q1))
        q1 = self.ln3(q1)
        return q1

    def get_all_actions_q(self, state, candidates, mask):
        candidates = candidates * mask.reshape(-1, 1)
        state = torch.repeat_interleave(state, candidates.shape[0], 0)
        scores = F.relu(self.ln1(torch.cat([state, candidates], dim=1)))
        scores = F.relu(self.ln2(scores))
        scores = self.ln3(scores)
        return scores.reshape(-1)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
    ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 250)
        self.ln2 = nn.Linear(250, 250)
        self.ln3 = nn.Linear(250, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, candidates, mask):
        state = torch.repeat_interleave(state, candidates.shape[0], 0)
        scores = F.relu(self.ln1(torch.cat([state, candidates], dim=1)))
        scores = F.relu(self.ln2(scores))
        prob = self.softmax(self.ln3(scores).reshape(-1)) * mask
        # sample action from prob
        dist = Categorical(prob / prob.sum())
        index = dist.sample()
        return index, candidates[index], prob[index]


class Scalar(nn.Module):
    def __init__(self, init_value) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant
