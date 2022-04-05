from munch import Munch
import copy
import numpy as np
from model.resnet import ResNet18, ResNet34, ResNet50
from model.mlp import MLP
from util.params import config

def build_model(args):
    if args.data == 'cmnist':
        debiased_model = MLP()
        biased_model = MLP()

        nets = Munch(biased=biased_model,
                     debiased=debiased_model)

        return nets

    else:
        pass

