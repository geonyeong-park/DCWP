from munch import Munch
import copy
import numpy as np
from model.resnet import ResNet18, ResNet34, ResNet50
from model.mlp import MLP, FC, SimCLRHead
from util.params import config

def build_model(args):
    if args.data == 'cmnist':
        debiased_model = MLP()
        biased_model = MLP()
        debiased_classifier = FC(MLP.last_dim, num_classes=10)
        biased_classifier = FC(MLP.last_dim, num_classes=10)
        head = SimCLRHead(MLP.last_dim, MLP.simclr_dim)

        nets = Munch(biased_F=biased_model,
                     debiased_F=debiased_model,
                     biased_C=biased_classifier,
                     debiased_C=debiased_classifier,
                     head=head)
        return nets

    else:
        pass

