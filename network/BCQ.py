import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,  # if normalized, set 1
                 phi=0.05  # max perturbation hyperparameter for BCQ
                 ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 250)
        self.ln2 = nn.Linear(250, 250)
        self.ln3 = nn.Linear(250, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.ln1(torch.cat([state, action], dim=1)))
        a = F.relu(self.ln2(a))
        a = self.phi * self.max_action * torch.tanh(self.ln3(a))
        return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim
                 ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(state_dim + action_dim, 250)
        self.ln2 = nn.Linear(250, 250)
        self.ln3 = nn.Linear(250, 1)

        self.ln4 = nn.Linear(state_dim + action_dim, 250)
        self.ln5 = nn.Linear(250, 250)
        self.ln6 = nn.Linear(250, 1)

    def forward(self, state, action):
        q1 = F.relu(self.ln1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.ln2(q1))
        q1 = self.ln3(q1)

        q2 = F.relu(self.ln4(torch.cat([state, action], dim=1)))
        q2 = F.relu(self.ln5(q2))
        q2 = self.ln6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.ln1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.ln2(q1))
        q1 = self.ln3(q1)
        return q1


class VAE(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 max_action,
                 ) -> None:
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + action_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], dim=1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # clamp for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        # when sampling from VAE, the latent vector is clipped
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(
                self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], dim=1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
