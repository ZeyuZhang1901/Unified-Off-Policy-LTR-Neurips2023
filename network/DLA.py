import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))

class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                torch.unsqueeze(
                    torch.zeros_like(
                        input_list[i]), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(
                torch.ones_like(input_list[i]), -1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    torch.cat(
                        click_feature, 1)))

        return torch.cat(output_propensity_list, 1)

class DNN(nn.Module):
    """The deep neural network model for learning to rank.

    This class implements a deep neural network (DNN) based ranking model. It's essientially a multi-layer perceptron network.

    """

    def __init__(self,feature_size):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """
        super(DNN, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
