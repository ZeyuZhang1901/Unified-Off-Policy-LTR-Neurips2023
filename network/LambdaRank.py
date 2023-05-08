import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DNN(nn.Module):
    def __init__(self, feature_size):
        super(DNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ln1 = nn.Linear(feature_size, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 1)
        # self.norm1 = nn.LayerNorm(feature_size)
        # self.norm2 = nn.LayerNorm(256)
        # self.norm3 = nn.LayerNorm(256)

    def forward(self, input_list):
        input_data = torch.cat(input_list, dim=0)
        input_data = input_data.to(dtype=torch.float32)
        input_data = input_data.to(self.device)

        # output_data = self.norm1(input_data)
        # output_data = F.relu(self.ln1(output_data))
        # output_data = self.norm2(output_data)
        # output_data = F.relu(self.ln2(output_data))
        # output_data = self.norm3(output_data)
        # output_data = self.ln3(output_data)

        output_data = F.relu(self.ln1(input_data))
        output_data = F.relu(self.ln2(output_data))
        output_data = self.ln3(output_data)

        output_shape = input_list[0].shape[0]
        return torch.split(output_data, output_shape, dim=0)
