import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class Actor(nn.Module):
    """Actor (Policy) model"""

    def __init__(
        self,
        action_dim,
        state_dim,
        num_node_list,  # list of num strings
    ):
        """Params:
        - `feature_size`: dimension of feature vector
        """
        super(Actor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = action_dim
        self.state_dim = state_dim

        num_node = self.action_dim + self.state_dim
        num_node_list = num_node_list + [1]
        self.sequential = nn.Sequential().to(dtype=torch.float32)
        for i in range(len(num_node_list)):
            next_num_node = int(num_node_list[i])
            # self.sequential.add_module(
            #     f"batchnorm{i+1}", nn.BatchNorm1d(num_node)
            # )
            self.sequential.add_module(
                f"linear{i+1}", nn.Linear(num_node, next_num_node)
            )
            if i != len(num_node_list) - 1:
                self.sequential.add_module(f"activation{i+1}", nn.ReLU())
            num_node = next_num_node

        # self.ln1 = nn.Linear(self.action_dim + self.state_dim, 256)
        # self.ln2 = nn.Linear(256, 256)
        # self.ln3 = nn.Linear(256, 256)
        # self.ln4 = nn.Linear(256, 256)
        # self.ln5 = nn.Linear(256, 1)

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

        # output_data = F.relu(self.ln1(input_data))
        # output_data = F.relu(self.ln2(output_data))
        # output_data = F.relu(self.ln3(output_data))
        # output_data = F.relu(self.ln4(output_data))
        # output_data = self.ln5(output_data)
        output_data = self.sequential(input_data)
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
        action_probs_mask = action_probs * masks
        action_probs_sum_mask = (
            torch.sum(action_probs_mask, dim=1, keepdim=True) == 0.0
        ).to(torch.float32)
        u_probs = (
            torch.rand_like(action_probs) + torch.ones_like(action_probs)
        ) * masks  # set random probs for invalid probs

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
        action_dim,
        state_dim,
        num_node_list,
    ):
        """Params:
        - `feature_size`: dimension of feature vector
        """
        super(Critic, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = action_dim
        self.state_dim = state_dim

        num_node = self.action_dim + self.state_dim
        num_node_list = num_node_list + [1]
        self.sequential = nn.Sequential().to(dtype=torch.float32)
        for i in range(len(num_node_list)):
            next_num_node = int(num_node_list[i])
            self.sequential.add_module(
                f"linear{i+1}", nn.Linear(num_node, next_num_node)
            )
            if i != len(num_node_list) - 1:
                self.sequential.add_module(f"activation{i+1}", nn.ReLU())
            num_node = next_num_node

        # self.ln1 = nn.Linear(self.action_dim + self.state_dim, 256)
        # self.ln2 = nn.Linear(256, 256)
        # self.ln3 = nn.Linear(256, 256)
        # self.ln4 = nn.Linear(256, 256)
        # self.ln5 = nn.Linear(256, 1)

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
        input_data = torch.cat([state, candidates], dim=1)

        # output_data = F.relu(self.ln1(input_data))
        # output_data = F.relu(self.ln2(output_data))
        # output_data = F.relu(self.ln3(output_data))
        # output_data = F.relu(self.ln4(output_data))
        # output_data = self.ln5(output_data)
        output_data = self.sequential(input_data)

        return output_data.reshape(-1, candidate_num)
