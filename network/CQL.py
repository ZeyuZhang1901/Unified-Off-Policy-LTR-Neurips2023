import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class Actor(nn.Module):
    """Actor (Policy) model"""

    def __init__(
        self,
        feature_size,
        state_type,
        list_length,
    ):
        """Params:
        - `feature_size`: dimension of feature vector
        """
        super(Actor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = feature_size
        if state_type == "pos" or state_type == "avg":
            self.state_dim = feature_size
        elif state_type == "pos_avg":
            self.state_dim = 2 * feature_size
        elif state_type == "pos_avg_rew":
            self.state_dim = 2 * feature_size + list_length
        elif state_type == "rew":
            self.state_dim = list_length
        

        self.ln1 = nn.Linear(self.action_dim + self.state_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        state,
        candidates,
    ):
        """Actor forward, input state and all actions, output prob of each action.
        Params:
        - `state (tensor float32)` shape [1, state_dim]
        - `candidates (tensor float32)` shape [candidate_num, action_dim]
        """
        state = state.to(torch.float32)
        candidates = candidates.to(torch.float32)

        candidate_num = int(candidates.shape[0] / state.shape[0])
        state = torch.repeat_interleave(state, candidate_num, dim=0)
        input_data = torch.cat([state, candidates], dim=1).to(self.device)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)
        # output_data += 1e-10  # in case
        output_data = output_data.reshape(-1, candidate_num)
        output_data = output_data - output_data.max(-1, keepdim=True).values

        return self.softmax(output_data, t=5)

    def evaluate(
        self,
        state,
        candidates,
        masks,
    ):
        """Action evaluation, input state and all actinos, output chosen action, action probs and log action probs
        - `state (tensor float32)` shape [1, state_dim]
        - `candidates (tensor float32)` shape [candidate_num, action_dim]
        """

        action_probs = self.forward(state, candidates)
        action_probs_mask = action_probs * masks.to(self.device)
        action_probs_sum_mask = (
            torch.sum(action_probs_mask, dim=1, keepdim=True) == 0.0
        ).to(
            torch.float32
        )  # incase sum of probs equals to zero
        u_probs = (
            torch.rand_like(action_probs) + torch.ones_like(action_probs)
        ) * masks.to(
            self.device
        )  # set random probs for invalid probs

        action_probs_valid = u_probs * action_probs_sum_mask + action_probs_mask * (
            1 - action_probs_sum_mask
        )
        action_probs = action_probs_valid / torch.sum(
            action_probs_valid, dim=1, keepdim=True
        )

        assert (
            torch.isnan(action_probs).any().float() == 0
        ), f"Nan appears in training, mask: {masks}\nprobs_mask: {action_probs_sum_mask}\nprobs: {action_probs_valid}"
        dist = Categorical(action_probs)
        index = dist.sample().to(self.device)

        z = action_probs == 0.0  ## mask out actions with prob=0
        z = z.float() * 1e-20
        log_action_probs = torch.log(action_probs + z)

        return index, action_probs, log_action_probs

    def get_action(
        self,
        state,
        candidates,
        masks,
    ):
        """Get action from all possible candidates"""
        action_probs = self.forward(state, candidates)
        action_probs_mask = action_probs * masks.to(self.device)
        action_probs_sum_mask = (
            torch.sum(action_probs_mask, dim=1, keepdim=True) == 0
        ).to(
            torch.float32
        )  # incase sum of probs equals to zero
        u_probs = (
            torch.rand_like(action_probs) + torch.ones_like(action_probs)
        ) * masks.to(
            self.device
        )  # set random probs for invalid probs

        action_probs_valid = u_probs * action_probs_sum_mask + action_probs_mask * (
            1 - action_probs_sum_mask
        )
        action_probs = action_probs_valid / torch.sum(
            action_probs_valid, dim=1, keepdim=True
        )

        assert (
            torch.isnan(action_probs).any().float() == 0
        ), f"Nan appears in validation, mask: {masks}\nprobs_mask: {action_probs_sum_mask}\nprobs: {action_probs_valid}"
        dist = Categorical(action_probs)
        index = dist.sample().to(self.device)

        return index

    def softmax(self, input, t=1.0):
        """t: temperature param, the larger, the stricter bound on different between max and min probs"""
        ex = torch.exp(input / t).clamp(min=1e-10)
        return ex / torch.sum(ex, dim=-1, keepdim=True)


class Critic(nn.Module):
    """Critic (Value) model."""

    def __init__(
        self,
        feature_size,
        state_type,
        list_length,
    ):
        """Params:
        - `feature_size`: dimension of feature vector
        """
        super(Critic, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = feature_size

        if state_type == "pos" or state_type == "avg":
            self.state_dim = feature_size
        elif state_type == "pos_avg":
            self.state_dim = 2 * feature_size
        elif state_type == "pos_avg_rew":
            self.state_dim = 2 * feature_size + list_length
        elif state_type == "rew":
            self.state_dim = list_length

        self.ln1 = nn.Linear(self.action_dim + self.state_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)

    def forward(
        self,
        state,
        candidates,
    ):
        """Critic Q function that maps (state, action) pairs to Q values"""
        state = state.to(torch.float32)
        candidates = candidates.to(torch.float32)

        candidate_num = int(candidates.shape[0] / state.shape[0])
        state = torch.repeat_interleave(state, candidate_num, dim=0)
        input_data = torch.cat([state, candidates], dim=1).to(self.device)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        return output_data.reshape(-1, candidate_num)
