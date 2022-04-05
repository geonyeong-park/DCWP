"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import io
import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile
import imageio
import argparse
import warnings
from pathlib import Path
from itertools import chain

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)

def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float))

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()

class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        assert len(data) > 1
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()

def moving_average_param(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

@torch.no_grad()
def debug_image(nets, args, inputs, step, num_images=32, denormalize=False):
    x_sup, y, idx = inputs.x_sup, inputs.y, inputs.index
    y_trg = inputs.y_trg
    c_src, c_trg = inputs.c_src, inputs.c_trg

    device = inputs.x_sup.device
    N = inputs.x_sup.size(0)

    # translate and reconstruct
    filename = ospj(args.result_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, x_sup, y, y_trg, c_src, c_trg, filename,
                              denormalize=denormalize, num_images=num_images)

@torch.no_grad()
def translate_and_reconstruct(nets, x_src, y_src, y_trg, c_src, c_trg, filename,
                              denormalize=False, num_images=32):
    x_src = x_src[:num_images]
    y_src = y_src[:num_images]
    y_trg = y_trg[:num_images]
    c_src = c_src[:num_images]
    c_trg = c_trg[:num_images]
    N, C, H, W = x_src.size()

    s_trg = nets.mapping_network(c_trg, y_trg)
    x_fake = nets.generator(x_src, s_trg)
    s_src = nets.mapping_network(c_src, y_src)
    x_recon = nets.generator(x_fake, s_src)

    x_concat = [x_src, x_fake, x_recon]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename, denormalize=denormalize)
    del x_concat

def save_image(x, ncol, filename, denormalize=False):
    if denormalize: x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


