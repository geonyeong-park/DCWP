import torch
import torch.nn as nn

class MLP(nn.Module):
    last_dim = 32
    simclr_dim = 16
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )

    def extract(self, x):
        x = x.view(x.size(0), -1) / 255
        return self.feature(x)

class FC(nn.Module):
    def __init__(self, last_dim, num_classes=10):
        super(FC, self).__init__()
        self.fc = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class SimCLRHead(nn.Module):
    def __init__(self, last_dim, simclr_dim=10):
        super(SimCLRHead, self).__init__()
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

    def forward(self, x):
        return self.simclr_layer(x)
