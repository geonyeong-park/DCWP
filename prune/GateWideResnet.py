import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prune.GateLayer import GateMLP, GateConv2d

__all__ = ['wrn']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = GateConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = GateConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and GateConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x, pruning=False, freeze=False):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x, pruning, freeze)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out, pruning, freeze)
        return torch.add(x if self.equalInOut else self.convShortcut(x, pruning, freeze), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.ModuleList(layers)

    def layer_forward(self, layer, x, pruning, freeze):
        for block in layer:
            x = block(x, pruning, freeze)
        return x

    def forward(self, x, pruning=False, freeze=False):
        return self.layer_forward(self.layer, x, pruning, freeze)

class GateWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(GateWideResNet, self).__init__()
        self.pruning = False
        self.freeze = False

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = GateConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = GateMLP(nChannels[3], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, GateConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GateMLP):
                m.bias.data.zero_()

    def _forward_impl(self, x, pruning=False, freeze=False, feature=False):
        out = self.conv1(x, pruning, freeze)
        out = self.block1(out, pruning, freeze)
        out = self.block2(out, pruning, freeze)
        out = self.block3(out, pruning, freeze)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)

        feature_ = out.view(-1, self.nChannels)
        out = self.fc(feature_)

        if feature:
            return out, feature_
        else:
            return out

    def forward(self, x, feature=False):
        return self._forward_impl(x, self.pruning, self.freeze, feature)

    def pruning_switch(self, turn_on=False):
        self.pruning = turn_on

    def freeze_switch(self, turn_on=False):
        self.freeze = turn_on


def wrn(depth, num_classes, widen_factor=1, dropRate=0.):
    """
    Constructs a Wide Residual Networks.
    """
    model = GateWideResNet(depth, num_classes, widen_factor, dropRate)
    return model

def GateWideResNet28_10(n_classes):
    model = wrn(28, n_classes, 10, 0.)
    return model

def GateWideResNet16_8(n_classes):
    model = wrn(16, n_classes, 8, 0.)
    return model
