import torch
import torch.nn as nn
from model.base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, num_classes=10):
        last_dim = 32
        simclr_dim = 16
        super(MLP, self).__init__(last_dim, num_classes, simclr_dim)

        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )

    def extract(self, x):
        x = x.view(x.size(0), -1)
        return self.feature(x)

