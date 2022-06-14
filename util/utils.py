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
import pickle

from tqdm import tqdm
import matplotlib.pyplot as plt

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

class ValidLogger(object):
    phase_token = ['ERM', 'prune', 'retrain', 'ratio']

    def __init__(self, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.fname = fname
        self.log = {
            'ERM': [],
            'prune': [],
            'retrain': [],
            'ratio': [] # Ratio of survived weights during pruning
        }

    def append(self, val, which='ERM'):
        self.log[which].append(val)

    def save(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.log, f)
            print(f'saved validation log in {self.fname}')

    def load(self):
        with open(self.fname, 'rb') as f:
            log = pickle.load(f)
            return log

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

def save_image(x, ncol, filename, denormalize=False):
    if denormalize: x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def plot_embedding(X, label, save_path):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    num_color = np.max(label) + 1
    cmap = plt.cm.get_cmap('rainbow', num_color)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    plt.scatter(X[:, 0], X[:, 1], c=label, cmap='rainbow')

    plt.xticks([]), plt.yticks([])
    #legend = ['source domain {}'.format(i+1) for i in range(min(d), max(d))]
    #legend[-1] = ['target domain']
    #plt.legend(legend)

    fig.savefig(save_path)
    plt.close('all')
