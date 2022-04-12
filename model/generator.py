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
    def __init__(self,
                 img_size=128,
                 num_channels=3,
                 num_classes=10,
                 channel_base=2048,
                 style_dim=128,
                 mapping_latent_dim=128,
                 class_embed_dim=128,
                 max_conv_dim=512,
                 sigmoid_output=False):
        # channel_base is high for bFFHQ
        super().__init__()
        self.img_size = img_size
        self.sigmoid_output = sigmoid_output
        self.decode = nn.ModuleList()
        if sigmoid_output:
            self.sigmoid = nn.Sigmoid()

        # Constant input for G
        self.const = torch.nn.Parameter(torch.randn([max_conv_dim, 4, 4]))

        # Mapping network
        self.mapping = MappingNetwork(mapping_latent_dim,
                                      class_embed_dim,
                                      style_dim,
                                      num_classes)

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 2

        dim_in = max_conv_dim
        for i in range(repeat_num):
            dim_out = min(channel_base // (2**(i+2)), max_conv_dim)
            self.decode.insert(
                0, AdainResBlk(dim_in, dim_out, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # to-rgb layer
        self.to_output = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, num_channels, 1, 1, 0))

    def forward(self, z, c):
        w = self.mapping(z, c)

        x = self.const
        x = x.unsqueeze(0).repeat([z.shape[0], 1, 1, 1])

        for block in self.decode:
            x = block(x, w)

        output = self.to_output(x)
        if self.sigmoid_output:
            output = self.sigmoid(output)
        return output


class MappingNetwork(nn.Module):
    def __init__(self,
                 mapping_latent_dim=128,
                 class_embed_dim=128,
                 style_dim=128,
                 num_classes=10,
                 num_layers=8):
        super().__init__()
        self.mapping_latent_dim = mapping_latent_dim

        features_list = [mapping_latent_dim + class_embed_dim] + \
            [style_dim] * (num_layers - 1) + [style_dim]
        layers = []

        self.embed = nn.Linear(num_classes, class_embed_dim)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layers += [nn.Linear(in_features, out_features),
                       nn.LeakyReLU()]

        self.fc = nn.Sequential(*layers)

    def forward(self, z, y):
        if self.mapping_latent_dim != 0:
            z = normalize_2nd_moment(z)
            y = normalize_2nd_moment(self.embed(y))
            x = torch.cat([z, y], dim=1)
        else:
            x = normalize_2nd_moment(self.embed(y))

        w = self.fc(x)
        return w


class Discriminator(nn.Module):
    """
    Discriminating real/fake and biased/debiased
    """
    def __init__(self,
                 img_size=256,
                 num_channels=3,
                 num_classes=10,
                 class_embed_dim=128,
                 cmap_dim=128,
                 channel_base=4096,
                 channel_max=512):
        super().__init__()
        self.img_size = img_size
        self.cmap_dim = cmap_dim
        self.img_resolution_log2 = int(np.log2(img_size))

        self.mapping = MappingNetwork(mapping_latent_dim=0,
                                      class_embed_dim=class_embed_dim,
                                      style_dim=cmap_dim,
                                      num_classes=num_classes)

        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}

        blocks = []
        blocks += [nn.Conv2d(num_channels, channels_dict[img_size], 3, 1, 1)]

        for res in self.block_resolutions:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            blocks += ResBlk(in_channels, out_channels, downsample=True)
        self.blocks = nn.Sequential(*blocks)

        self.real_fake_classifier = DiscriminatorEpilogue(out_channels, cmap_dim=class_embed_dim)
        self.bias_classifier = DiscriminatorEpilogue(out_channels, cmap_dim=class_embed_dim)

    def forward(self, x, c):
        cmap = self.mapping(None, c)
        h = self.blocks(x)

        out_bias = self.bias_classifier(h)
        out_real = self.real_fake_classifier(h)

        out_bias = (out_bias * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        out_real = (out_real * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out_bias, out_real


class DiscriminatorEpilogue(nn.Module):
    def __init__(self,
                 in_channels,
                 resolution=4,
                 cmap_dim=128):
        super().__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2)
        ])

        self.fc = nn.Sequential(*[
            nn.Linear(in_channels * (resolution ** 2), in_channels),
            nn.LeakyReLU(0.2)
        ])

        self.out = nn.Linear(in_channels, cmap_dim)

    def forward(self, h):
        x = self.conv(h)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        return x


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

