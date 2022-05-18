import torch
import torch.nn.functional as F
import torch.nn as nn
from prune.GumbelSigmoid import GumbelSigmoidMask

class GateMLP(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(GateMLP, self).__init__(in_features, out_features, bias=bias)
        self.mask = GumbelSigmoidMask(self.weight.shape)

    def forward(self, input, pruning=False, freeze=False):
        mask = None
        if pruning:
            mask = self.mask.sample(hard=True)

        if freeze:
            mask = self.mask.fix_mask_after_pruning()

        if mask is not None:
            return F.linear(input, self.weight*mask.to(input.device), self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class GateConv2d(nn.Conv2d):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=0, bias=True,
                 dilation=1, groups=1):
        super(GateConv2d, self).__init__(in_features, out_features, kernel_size,
                                         stride=stride, padding=padding, bias=bias,
                                         dilation=dilation, groups=groups)
        self.mask = GumbelSigmoidMask(self.weight.shape)

    def forward(self, input, pruning=False, freeze=False):
        mask = None
        if pruning:
            mask = self.mask.sample(hard=True)

        if freeze:
            mask = self.mask.fix_mask_after_pruning()

        if mask is not None:
            return F.conv2d(input, self.weight*mask.to(input.device), self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

