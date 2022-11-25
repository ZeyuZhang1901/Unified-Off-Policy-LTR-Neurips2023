import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from ranker.AbstractRanker import AbstractRanker
from network.DQN import DQN
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'chosen', 'qid'))

class DoubleDQNRanker(AbstractRanker):
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

    def update_policy(self, memory, dataset):
        trainsitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*trainsitions))
        states = torch.cat(batch.state).to(self.device)
        # state_batch = torch.zeros_like(torch.cat(batch.state), dtype=torch.float32).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        nextstates = torch.cat(batch.next_state).to(self.device)
        # next_state_batch = torch.zeros_like(torch.cat(batch.next_state), dtype=torch.float32).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        dones = torch.cat(batch.done).to(self.device)

        Q_s_a = self.q(states, actions)
        with torch.no_grad():
            Q_nexts_nexta = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(self.device)
            for i in range(self.batch_size):
                if batch.done[i] == True:
                    continue
                candidates = torch.tensor(dataset.get_all_features_by_query(batch.qid[i])).to(self.device).to(torch.float32)
                docid = dataset.get_docid_by_query_and_feature(batch.qid[i], actions[i])
                batch.chosen[i][docid] = False
                candidates = candidates[batch.chosen[i]]
                next_action = self.selectAction(state=nextstates[i].view(1,-1), candidates=candidates)
                next_action = torch.tensor(next_action).reshape(1,-1).to(self.device)
                Q_nexts_nexta[i,0] = self.target_q(state=nextstates[i].view(1,-1), action=next_action)
            Q_nexts_nexta = rewards + (1-dones) * self.discount * Q_nexts_nexta
        
        loss = F.mse_loss(Q_s_a, Q_nexts_nexta)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network soft update
        for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
           target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)

        return Q_s_a.mean().item(), Q_nexts_nexta.mean().item(), loss.item()

    def get_query_result_list(self, dataset, query):
        candidates = dataset.get_all_features_by_query(query).astype(np.float32)
        docid_list = dataset.get_candidate_docids_by_query(query)
        ndoc = len(docid_list)
        ranklist = np.zeros(ndoc, dtype=np.int32)

        state = np.zeros(self.state_dim, dtype=np.float32)
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        for pos in range(ndoc):
            # state
            state = next_state
            # action 
            action = self.selectAction(state=state.reshape(1,-1), candidates=candidates)
            # reward
            docid = dataset.get_docid_by_query_and_feature(query, action)
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            ranklist[pos] = docid
            # next state
            next_state[:self.action_dim] = action + pos*state[:self.action_dim]/(pos+1)
            if self.action_dim+pos < self.state_dim:
                next_state[self.action_dim + pos] = 1/np.log2(pos+2) if relevance > 0 else 0
            # delete chosen doc in candidates
            for i in range(candidates.shape[0]):
                if np.array_equal(candidates[i], action):
                    candidates = np.delete(candidates, i, axis=0)
                    break
        
        return ranklist

    def get_all_query_result_list(self, dataset):
        query_result_list = {}
        for query in dataset.get_all_querys():
            query_result_list[query] = self.get_query_result_list(dataset, query)

        return query_result_list

    def selectAction(self,
                    state,
                    candidates):
        with torch.no_grad():
            if type(state)==np.ndarray:
                state = torch.from_numpy(state)
            if type(candidates)==np.ndarray:
                candidates = torch.from_numpy(candidates) 
            state = state.expand(candidates.shape[0], -1).to(self.device)
            candidates = candidates.to(self.device)
            scores = self.q(state, candidates)
            # debug = torch.max(scores, 0)[1]
            # debug2 = candidates[debug]
            # debug3 = debug2.squeeze(0)
            return candidates[torch.max(scores, 0)[1]].squeeze(0).cpu().numpy()

    def getTargetMaxValue(self,
                        state,
                        candidates):

        scores = self.target_q(state.expand(candidates.shape[0], -1), candidates)
        return torch.max(scores, 0)[0].item()
