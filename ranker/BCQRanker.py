import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from network.DQN import DQN
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'chosen', 'qid'))

class BCQRanker(AbstractRanker):
    def __init__(self,
                state_dim,
                action_dim,
                lr=1e-3,
                batch_size = 256,
                discount = 0.9,
                tau = 0.005  # soft update rate
                ):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.q = DQN(state_dim+action_dim).to(self.device)
        self.target_q = copy.deepcopy(self.q).to(self.device)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)