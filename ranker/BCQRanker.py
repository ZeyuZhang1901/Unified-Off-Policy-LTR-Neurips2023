import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from network.BCQ import Actor, Critic, VAE
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action',
                        'next_state', 'reward', 'done', 'chosen', 'qid'))


class BCQRanker(AbstractRanker):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 batch_size,
                 lr=1e-3,
                 discount=0.9,
                 tau=0.005,
                 lmbda=0.75,
                 phi=0.05):
        latent_dim = action_dim * 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # actor part
        self.actor = Actor(state_dim, action_dim,
                           max_action, phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)
        # critic part
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)
        # vae part
        self.vae = VAE(state_dim, action_dim, latent_dim,
                       max_action).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda

    def update_policy(self, memory):
        trainsitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*trainsitions))
        states = torch.cat(batch.state).to(self.device)
        # state_batch = torch.zeros_like(torch.cat(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        nextstates = torch.cat(batch.next_state).to(self.device)
        # next_state_batch = torch.zeros_like(torch.cat(batch.next_state), dtype=torch.float32).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        dones = torch.cat(batch.done).to(self.device)

        # VAE training
        recon, mean, std = self.vae(states, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) -
                          mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic training
        with torch.no_grad():
            nextstates = torch.repeat_interleave(nextstates, 10, 0)
            # compute value of perturbed actions sampled form the VAE
            target_Q1, target_Q2 = self.critic_target(
                nextstates, self.actor_target(nextstates, self.vae.decode(nextstates)))
            # soft clipped Double-Q-learning
            target_Q = self.lmbda * \
                torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * \
                torch.max(target_Q1, target_Q2)
            # take max over each action sampled form the VAE
            target_Q = target_Q.reshape(
                self.batch_size, -1).max(1)[0].reshape(-1, 1)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Pertubation Model (actor) training
        sampled_actions = self.vae.decode(states)
        perturbed_actions = self.actor(states, sampled_actions)
        actor_loss = -self.critic.q1(states, perturbed_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # targets update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        return current_Q1.mean().item(), \
            current_Q2.mean().item(), \
            target_Q1.mean().item(), \
            target_Q2.mean().item(), \
            critic_loss.item(),\
            actor_loss.item()

    def selectAction(self,
                     state,
                     candidates):
        with torch.no_grad():
            if type(state) == np.ndarray:
                state = torch.from_numpy(state)
            if type(candidates) == np.ndarray:
                candidates = torch.from_numpy(candidates)
            state = state.expand(candidates.shape[0], -1).to(self.device)
            candidates = candidates.to(self.device)
            scores = self.critic.q1(state, candidates)
            return candidates[torch.max(scores, 0)[1]].squeeze(0).cpu().numpy()

    def get_query_result_list(self, dataset, query):
        candidates = dataset.get_all_features_by_query(
            query).astype(np.float32)
        docid_list = dataset.get_candidate_docids_by_query(query)
        ndoc = len(docid_list)
        ranklist = np.zeros(ndoc, dtype=np.int32)

        state = np.zeros(self.state_dim, dtype=np.float32)
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        for pos in range(ndoc):
            # state
            state = next_state
            # action
            action = self.selectAction(
                state=state.reshape(1, -1), candidates=candidates)
            # reward
            docid = dataset.get_docid_by_query_and_feature(query, action)
            relevance = dataset.get_relevance_label_by_query_and_docid(
                query, docid)
            ranklist[pos] = docid
            # next state
            next_state[:self.action_dim] = action + \
                pos*state[:self.action_dim]/(pos+1)
            if self.action_dim+pos < self.state_dim:
                next_state[self.action_dim + pos] = 1 / \
                    np.log2(pos+2) if relevance > 0 else 0
            # delete chosen doc in candidates
            for i in range(candidates.shape[0]):
                if np.array_equal(candidates[i], action):
                    candidates = np.delete(candidates, i, axis=0)
                    break

        return ranklist

    def get_all_query_result_list(self, dataset):
        query_result_list = {}
        for query in dataset.get_all_querys():
            query_result_list[query] = self.get_query_result_list(
                dataset, query)

        return query_result_list

    def restore_ranker(self, path):
        print('restore ranker start!')
        torch.save(self.actor.state_dict(), path+'actor.pt')
        torch.save(self.actor_target.state_dict(), path+'actor_target.pt')
        torch.save(self.critic.state_dict(), path+'critic.pt')
        torch.save(self.critic_target.state_dict(), path+'critic_target.pt')
        torch.save(self.vae.state_dict(), path+'vae.pt')
        print('restore ranker finish!')

    def load_ranker(self, path):
        print('load ranker start!')
        self.actor.load_state_dict(torch.load(path+'actor.pt'))
        self.actor_target.load_state_dict(torch.load(path+'actor_target.pt'))
        self.critic.load_state_dict(torch.load(path+'critic.pt'))
        self.critic_target.load_state_dict(torch.load(path+'critic_target.pt'))
        self.vae.load_state_dict(torch.load(path+'vae.pt'))
        print('load ranker finish!')
