import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
    ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)
        self.norm1 = nn.LayerNorm(state_dim + action_dim)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)

    def forward(self, state, action):
        q1 = self.norm1(torch.cat([state, action], dim=1))
        q1 = F.relu(self.ln1(q1))
        q1 = self.norm2(q1)
        q1 = F.relu(self.ln2(q1))
        q1 = self.norm3(q1)
        q1 = self.ln3(q1)
        return q1

    def get_all_actions_q(self, state, candidates, mask):
        candidates = candidates * mask.reshape(-1, 1)
        state = torch.repeat_interleave(state, candidates.shape[0], 0)
        scores = self.norm1(torch.cat([state, candidates], dim=1))
        scores = F.relu(self.ln1(scores))
        scores = self.norm2(scores)
        scores = F.relu(self.ln2(scores))
        scores = self.norm3(scores)
        scores = self.ln3(scores)
        return scores.reshape(-1)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
    ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)
        self.norm1 = nn.LayerNorm(state_dim + action_dim)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, candidates, mask):
        state = torch.repeat_interleave(state, candidates.shape[0], 0)
        scores = self.norm1(torch.cat([state, candidates], dim=1))
        scores = F.relu(self.ln1(scores))
        scores = self.norm2(scores)
        scores = F.relu(self.ln2(scores))
        scores = self.norm3(scores)
        scores = self.ln3(scores).reshape(-1)
        prob = F.softmax(scores, dim=0) * mask
        # sample action from prob
        if not prob.sum() == 0:  # in case no action to choose
            prob = prob / prob.sum()
            # t = prob.detach().cpu().numpy()
            # print(t)
            index = np.random.choice(
                list(range(len(prob))), p=prob.detach().cpu().numpy()
            )
            return index, candidates[index], prob[index]
        else:
            print(mask)
            if mask.sum() != 0:
                print("debug start!")


class Scalar(nn.Module):
    def __init__(self, init_value) -> None:
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant
