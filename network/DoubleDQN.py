import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDQN(nn.Module):
    r"""Q-network implementation

    `Input`: state(shape=[batch_size, STATE_DIM])
            action(shape=[batch_size, ACTION_DIM])
    `Output`: Q value(shape=[batch_size, 1])"""

    def __init__(
        self,
        feature_size,
        state_type,
        list_length,
    ) -> None:
        super(DoubleDQN, self).__init__()
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

        # self.ln1 = nn.Linear(self.state_dim + self.action_dim, 256)
        # self.ln2 = nn.Linear(256, 256)
        # self.ln3 = nn.Linear(256, 1)
        # self.norm1 = nn.LayerNorm(self.state_dim + self.action_dim)
        # self.norm2 = nn.LayerNorm(256)
        # self.norm3 = nn.LayerNorm(256)

        self.ln1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)

        # self.ln = nn.Linear(self.state_dim + self.action_dim, 1)

    def forward_current(
        self,
        states,
        actions,
        batch_size,
    ):

        states = states.to(torch.float32)
        actions = actions.to(torch.float32)
        input_data = torch.cat([states, actions], dim=1)

        # forward
        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        return torch.split(output_data, batch_size, dim=0)

    def forward_next(
        self,
        valid_mask,  # bool type, indicate valid document ([batch_size, candidate_num])
        # cum_input_feature_list,
        position_input_list,
        # reward_input_list,
        candidate_list,
    ):
        input_position = torch.cat(position_input_list, dim=0).to(torch.float32)
        # input_reward = torch.cat(reward_input_list, dim=0).to(torch.float32)
        # input_cum_feature = torch.cat(cum_input_feature_list, dim=0).to(torch.float32)

        # input_state = torch.cat(
        #     [input_cum_feature, input_position, input_reward], dim=1
        # )
        # input_state = torch.cat([input_cum_feature, input_position], dim=1)
        input_state = input_position

        candidate_num = candidate_list[0].shape[0]
        input_state = torch.repeat_interleave(input_state, candidate_num, dim=0)
        input_candidate = torch.cat(candidate_list, dim=0).to(torch.float32)
        input_candidate = torch.cat([input_candidate] * candidate_num, dim=0)
        input_data = torch.cat([input_state, input_candidate], dim=1).to(self.device)
        # input_data = input_candidate.to(self.device)

        # forward
        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        # output_data = self.ln(input_data)

        output_shape = len(candidate_list) * candidate_num  # batch_size * rank_size
        raw_scores_list = torch.split(output_data, output_shape, dim=0)
        value_list, index_list = [], []
        valid_mask = valid_mask.to(torch.float32)
        for i in range(len(raw_scores_list)):
            scores = raw_scores_list[i].reshape(
                -1, candidate_num
            )  # batch_size * candidate_num
            scores = valid_mask * scores + (1 - valid_mask) * -1e6  # mask invalid docs
            mask = torch.cat([torch.zeros(i), torch.ones(candidate_num - i)], dim=0).to(
                self.device
            )  # mask out chosen documents
            scores = torch.where(
                torch.stack([mask] * scores.shape[0]) > 0,
                scores,
                torch.ones_like(scores) * -1e8,  # for punishment
            )
            value, index = torch.max(scores, dim=-1, keepdim=True)
            value_list.append(value)
            index_list.append(index)

        return value_list, index_list

    def forward_next(
        self,
        states,
        actions,
        candidate_num,
        batch_size,
    ):

        states = states.to(torch.float32)
        actions = actions.to(torch.float32)
        input_data = torch.cat([states, actions], dim=1)

        # forward
        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        output_shape = candidate_num * batch_size
        raw_scores_list = torch.split(output_data, output_shape, dim=0)
        index_list = []
        for i in range(len(raw_scores_list)):
            scores = raw_scores_list[i].reshape(-1, candidate_num)
            mask = torch.cat([torch.zeros(i), torch.ones(candidate_num - i)], dim=0)
            masks = torch.stack([mask] * batch_size).to(self.device)
            scores = scores * masks + torch.ones_like(scores) * -1e8 * (1 - masks)
            index_list.append(torch.max(scores, dim=-1, keepdim=True).indices) 
        return index_list

    def forward(
        self,
        states,
        candidates,
        masks,
        candidate_num,
    ):

        states = states.to(torch.float32)
        candidates = candidates.to(torch.float32)
        input_data = torch.cat([states, candidates], dim=1)

        # forward
        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        scores = output_data.reshape(-1, candidate_num)  # batch_size * candidate_num
        scores = scores * masks.to(self.device) + (torch.ones_like(scores) * -1e8) * (
            1 - masks.to(self.device)
        )
        values, index = torch.max(scores, dim=1, keepdim=True)
        return values, index.flatten()
