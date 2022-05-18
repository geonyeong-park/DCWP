import torch
import torch.nn.functional as F
import torch.nn as nn
from prune.GateLayer import GateMLP, GateConv2d

class GateCNN(nn.Module):
    # For cmnist only
    def __init__(self):
        super().__init__()
        self.pruning = False
        self.freeze = False

        self.conv1 = GateConv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = GateConv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = GateConv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = GateMLP(256, 10)  # We only use this simple CNN on cmnist

    def forward(self, x, feature=False):
        out = self.conv1(x, self.pruning, self.freeze)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, self.pruning, self.freeze)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, self.pruning, self.freeze)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.avgpool(out)
        feature_ = torch.flatten(out, 1)
        logit = self.linear(feature_, self.pruning, self.freeze)

        if feature:
            return logit, feature_
        else:
            return logit

    def extract(self, x):
        _, feature = self.forward(x, feature=True)
        return feature

    def pruning_switch(self, turn_on=False):
        self.pruning = turn_on

    def freeze_switch(self, turn_on=False):
        self.freeze = turn_on


class GateFCN(nn.Module):
    # For cmnist only. Some baselines use FCN in cmnist task
    def __init__(self):
        super(GateFCN, self).__init__()
        self.pruning = False
        self.freeze = False

        self.linear1 = GateMLP(3*32*32, 100)
        self.linear2 = GateMLP(100, 100)
        self.linear3 = GateMLP(100, 32)
        self.linear4 = GateMLP(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x, feature=False):
        # x is already normalized to [0,1], but the original implementation of LfF or
        # FeatureSwap divide it once again.
        x = x.view(x.size(0), -1) / 255
        out = self.linear1(x, self.pruning, self.freeze)
        out = self.relu(out)
        out = self.linear2(out, self.pruning, self.freeze)
        out = self.relu(out)
        out = self.linear3(out, self.pruning, self.freeze)
        out = self.relu(out)

        feature_ = torch.flatten(out, 1)
        logit = self.linear4(feature_, self.pruning, self.freeze)

        if feature:
            return logit, feature_
        else:
            return logit

    def extract(self, x):
        _, feature = self.forward(x, feature=True)
        return feature

    def pruning_switch(self, turn_on=False):
        self.pruning = turn_on

    def freeze_switch(self, turn_on=False):
        self.freeze = turn_on
