import torch
import torch.nn as nn


class DQN(nn.Module):
    r'''Q-network implementation

        `Input`: state(shape=[batch_size, STATE_DIM])
                action(shape=[batch_size, ACTION_DIM])
        `Output`: Q value(shape=[batch_size, 1])'''

    def __init__(self,
                 num_feature
                 ) -> None:
        super().__init__()
        self.ln1 = nn.Linear(num_feature, 256, dtype=torch.float32)
        self.ln2 = nn.Linear(256, 256, dtype=torch.float32)
        self.output = nn.Linear(256, 1, dtype=torch.float32)
        self.net = nn.Sequential(
            self.ln1,
            nn.ReLU(),
            self.ln2,
            nn.ReLU(),
            self.output,
        )
        self.net.apply(init_weights)

    def forward(self,
                state,
                action):
        # return self.net(torch.cat((state, action), dim=1)).clamp(-100,100)
        return self.net(torch.cat((state, action), dim=1))


def init_weights(m):  # initial weights mannually
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
