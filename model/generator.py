"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mlp import MLP


"""
Block architecture
"""

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

"""
Networks
"""

class ResGenerator(nn.Module):
    def __init__(self, img_size=128, dim_in=32, style_dim=64, max_conv_dim=512,
                 num_channels=3, sigmoid_output=False):
        super().__init__()
        self.img_size = img_size
        self.sigmoid_output = sigmoid_output
        self.from_input = nn.Conv2d(num_channels, dim_in, 3, 1, 1)

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_output = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, num_channels, 1, 1, 0))

        if sigmoid_output:
            self.sigmoid = nn.Sigmoid()

        # down/up-sampling blocks
        repeat_num = max(int(np.log2(img_size)) - 4, 2)

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(1):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))  # stack-like

    def forward(self, x, s):
        # content will be barely used in ResGenerator
        x = self.from_input(x)
        cache = {}

        for block in self.encode:
            x = block(x)

        for block in self.decode:
            x = block(x, s)

        output = self.to_output(x)
        if self.sigmoid_output:
            output = self.sigmoid(output)

        return output

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        dim_in = style_dim // 2
        layers += [
            nn.Conv2d(num_domains, dim_in, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2)
        ]

        repeat_num = int(np.log2(latent_dim)) - 1

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, 512)
            layers += [
                nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ]
            dim_in = dim_out
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(dim_out, dim_out),
                                            nn.ReLU(),
                                            nn.Linear(dim_out, dim_out),
                                            nn.ReLU(),
                                            nn.Linear(dim_out, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        h = h.view(h.size(0), h.size(1))
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    """Could be used in bFFHQ dataset. Almost same as MappingNetwork"""
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512,
                 num_channels=3):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(num_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class MLPDiscriminator(MLP):
    """
    Discriminating real/fake and (biased) class (For ColorMNIST only)
    Debiased discriminator only discriminates real/fake
        - It is implemented in MLP.py (For ColorMNIST) or ResNet.py

    #TODO: do BN?
    """
    def __init__(self):
        super(MLPDiscriminator, self).__init__(num_classes=10)
        self.real_fake_classifier = nn.Linear(100, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.feature(x)

        out_cls = self.classifier(h)
        out_real = self.real_fake_classifier(h)
        return out_real, out_cls.view(out_cls.size(0), -1)


class ConvDiscriminator(nn.Module):
    """
    Discriminating real/fake and (biased) class (For all datasets except ColorMNIST)

    """
    def __init__(self, img_size=256, dim_in=32, num_domains=10, max_conv_dim=512,
                 num_channels=3):
        super().__init__()
        self.img_size = img_size

        blocks = []
        blocks += [nn.Conv2d(num_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for idx in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += ResBlk(dim_in, dim_out, downsample=True)
            dim_in = dim_out

        self._make_layers(dim_out, num_domains)

    def _make_layers(self, dim_out, num_domains):
        self.real_fake_classifier = nn.Conv2d(dim_out, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.classifier = nn.Conv2d(dim_out, num_domains, kernel_size=4, bias=False)

    def forward(self, x):
        h = self.blocks(x)

        out_cls = self.classifier(h)
        out_real = self.real_fake_classifier(h)
        return out_real.view(out_real.size(0), -1), out_cls.view(out_cls.size(0), -1)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

