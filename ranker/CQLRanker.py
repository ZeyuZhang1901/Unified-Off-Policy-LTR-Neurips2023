import torch
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
        lr=1e-3,
        discount=0.9,
        tau=0.005,
        cql_min_q_weight=1,  # param before punishment item
        use_cql=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # actor part: get action and probability distribution
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        # critic part: get scores for each state-action pair
        self.qf1 = Critic(state_dim, action_dim).to(self.device)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.qf2 = Critic(state_dim, action_dim).to(self.device)
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
        self.tau = tau  # soft target update rate
        self.train_set = train_set
        self.use_cql = use_cql  # if false, degenerate to SAC
        self.cql_min_q_weight = cql_min_q_weight

    def update_policy(self, memory):
        # get batch
        trainsitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*trainsitions))
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        nextstates = torch.cat(batch.next_state).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        dones = torch.cat(batch.done).to(self.device)

        # actor get new action and prob for each state in batch
        new_actions = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        new_indexs = torch.zeros(self.batch_size, dtype=torch.int).to(self.device)
        new_probs = torch.zeros(self.batch_size, dtype=torch.float32).to(self.device)
        for i in range(self.batch_size):
            candidates = (
                torch.from_numpy(self.train_set.get_all_features_by_query(batch.qid[i]))
                .clone()
                .to(torch.float32)
                .to(self.device)
            )
            chosen = batch.chosen[i].to(self.device)
            new_indexs[i], new_actions[i], new_probs[i] = self.actor(
                states[i].unsqueeze(0), candidates, chosen
            )

        # tuning alpha (automatic entropy tuning)
        alpha_loss = -(self.log_alpha() * torch.log(new_probs + 1e-9).detach()).mean()
        alpha = self.log_alpha().exp()

        # actor loss
        q_new_actions = torch.min(
            self.qf1(states, new_actions),
            self.qf2(states, new_actions),
        )
        actor_loss = (alpha * torch.log(new_probs + 1e-9) - q_new_actions).mean()

        # critic loss
        q1_pred = self.qf1(states, actions)
        q2_pred = self.qf2(states, actions)

        new_next_actions = []
        for i in range(self.batch_size):
            candidates = (
                torch.from_numpy(self.train_set.get_all_features_by_query(batch.qid[i]))
                .clone()
                .to(torch.float32)
                .to(self.device)
            )
            chosen = batch.chosen[i].to(self.device)
            index = self.train_set.get_docid_by_query_and_feature(
                batch.qid[i], actions[i]
            )
            chosen[index] = False
            if not dones[i]:
                _, new_next_action, _ = self.actor(
                    nextstates[i].unsqueeze(0), candidates, chosen
                )
                new_next_actions.append(new_next_action.unsqueeze(0))
        new_next_actions = torch.cat(new_next_actions).to(self.device)

        nextstates = nextstates[dones.sum(dim=1) == 0]
        rewards = rewards[dones.sum(dim=1) == 0]
        target_q_values = torch.min(
            self.target_qf1(nextstates, new_next_actions),
            self.target_qf2(nextstates, new_next_actions),
        )
        td_target = rewards + self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred[dones.sum(dim=1) == 0], td_target.detach())
        qf2_loss = F.mse_loss(q2_pred[dones.sum(dim=1) == 0], td_target.detach())

        # add CQL term to qf loss
        if not self.use_cql:
            qf_loss = qf1_loss + qf2_loss
        else:
            cql_cat_q1s = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(
                self.device
            )
            cql_cat_q2s = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(
                self.device
            )
            for i in range(self.batch_size):
                candidates = torch.tensor(
                    self.train_set.get_all_features_by_query(batch.qid[i]),
                    dtype=torch.float32,
                ).to(self.device)
                state_q1s = self.qf1.get_all_actions_q(
                    states[i].unsqueeze(0), candidates, batch.chosen[i].to(self.device)
                )
                state_q2s = self.qf2.get_all_actions_q(
                    states[i].unsqueeze(0), candidates, batch.chosen[i].to(self.device)
                )
                cql_cat_q1s[i] = torch.logsumexp(state_q1s, 0)
                cql_cat_q2s[i] = torch.logsumexp(state_q2s, 0)
            cql_qf1_diff = (cql_cat_q1s - q1_pred).mean()
            cql_qf2_diff = (cql_cat_q2s - q2_pred).mean()
            # punishment items
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight
            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        # optimization
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # target net update
        for param, target_param in zip(
            self.qf1.parameters(), self.target_qf1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.qf2.parameters(), self.target_qf2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return (
            q1_pred.mean().item(),
            q2_pred.mean().item(),
            target_q_values.mean().item(),
            actor_loss.item(),
            qf_loss.item(),
        )

    def selectAction(self, state, candidates, mask):
        with torch.no_grad():
            if type(state) == np.ndarray:
                state = torch.from_numpy(state)
            if type(candidates) == np.ndarray:
                candidates = torch.from_numpy(candidates)
            candidates = candidates.to(self.device)
            index, action, _ = self.actor(state, candidates, mask)
            return index, action

    def get_query_result_list(self, dataset, query):
        candidates = dataset.get_all_features_by_query(query).astype(np.float32)
        docid_list = dataset.get_candidate_docids_by_query(query)
        end_pos = int((self.state_dim - self.action_dim) / 2)
        ndoc = len(docid_list) if len(docid_list) < end_pos else end_pos
        ranklist = np.zeros(ndoc, dtype=np.int32)

        state = np.zeros(self.state_dim, dtype=np.float32)
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        mask = torch.ones(candidates.shape[0], dtype=bool).to(self.device)
        for pos in range(ndoc):
            # state
            state = next_state
            # action
            docid, action = self.selectAction(
                state=torch.from_numpy(state).reshape(1, -1).to(self.device),
                candidates=torch.from_numpy(candidates).to(self.device),
                mask=mask,
            )
            action = action.cpu().numpy()
            # reward
            relevance = dataset.get_relevance_label_by_query_and_docid(
                query, int(docid)
            )
            ranklist[pos] = int(docid)
            # next state
            next_state[: self.action_dim] = action + pos * state[: self.action_dim] / (
                pos + 1
            )
            if pos < end_pos:
                next_state[self.action_dim + pos] = (
                    1 / np.log2(pos + 2) if relevance > 0 else 0
                )
                next_state[self.action_dim + end_pos + pos] = 1
                if pos > 0:
                    next_state[self.action_dim + end_pos + pos - 1] = 0
            # renew mask
            mask[docid] = False

        return ranklist

    def get_all_query_result_list(self, dataset):
        query_result_list = {}
        for query in dataset.get_all_querys():
            query_result_list[query] = self.get_query_result_list(dataset, query)
        return query_result_list

    def restore_ranker(self, path):
        print("restore ranker start!")
        torch.save(self.actor.state_dict(), path + "actor.pt")
        torch.save(self.qf1.state_dict(), path + "qf1.pt")
        torch.save(self.qf2.state_dict(), path + "qf2.pt")
        torch.save(self.target_qf1.state_dict(), path + "target_qf1.pt")
        torch.save(self.target_qf2.state_dict(), path + "target_qf2.pt")
        print("restore ranker finish!")

    def load_ranker(self, path):
        print("load ranker start!")
        self.actor.load_state_dict(torch.load(path + "actor.pt"))
        self.qf1.load_state_dict(torch.load(path + "qf1.pt"))
        self.qf2.load_state_dict(torch.load(path + "qf2.pt"))
        self.target_qf1.load_state_dict(torch.load(path + "target_qf1.pt"))
        self.target_qf2.load_state_dict(torch.load(path + "target_qf2.pt"))
        print("load ranker finish!")
