import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

class DDQN(nn.Module):
    def __init__(
        self,
        action_dim,
        state_dim,
    ):
        """Params:
        - `feature_size`: dimension of feature vector
        """
        super(DDQN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.ln1 = nn.Linear(self.action_dim + self.state_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)

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
        output_data = output_data.reshape(-1, candidate_num)

        return output_data