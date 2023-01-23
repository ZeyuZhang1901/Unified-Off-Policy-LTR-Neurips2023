import torch
import torch.nn as nn
import torch.nn.functional as F


class Bandit(nn.Module):
    def __init__(
        self,
        feature_size,
    ) -> None:
        super(Bandit, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = feature_size

        self.ln1 = nn.Linear(self.action_dim, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)
        # self.norm1 = nn.LayerNorm(self.action_dim)
        # self.norm2 = nn.LayerNorm(256)
        # self.norm3 = nn.LayerNorm(256)

        # self.ln = nn.Linear(self.action_dim, 1)

    def forward_current(self, input_feature_list):
        input_feature = torch.cat(input_feature_list, dim=0).to(torch.float32)
        input_data = input_feature.to(self.device)

        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        # return self.ln(input_data)
        return output_data

    def forward(self, candidates_list):
        candidate_num = candidates_list[0].shape[0]
        input_candidate = torch.cat(candidates_list, dim=0).to(torch.float32)
        input_data = input_candidate.to(self.device)

        # output_data = self.ln(input_data)

        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        return output_data.reshape(-1, candidate_num)
