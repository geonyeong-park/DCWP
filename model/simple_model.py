import torch
import torch.nn as nn


class CNN(nn.Module):
    # For cmnist only
    def __init__(self):
        super().__init__()
        self.last_dim = 256

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def extract(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.avgpool(out)
        feature = torch.flatten(out, 1)
        return feature


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.last_dim = 16

        self.feature = nn.Sequential(
            nn.Linear(3*32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )

    def extract(self, x):
        # x is already normalized to [0,1], but the original implementation of LfF or
        # FeatureSwap divide it once again.
        x = x.view(x.size(0), -1) / 255
        return self.feature(x)

class FC(nn.Module):
    def __init__(self, last_dim, num_classes=10):
        super(FC, self).__init__()
        self.fc = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

