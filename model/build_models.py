from munch import Munch
import copy
import numpy as np
from model.simple_model import CNN, MLP, FC
from prune.GateSimpleModel import GateCNN, GateFCN

from util.params import config

def build_model(args):
    if args.mode == 'prune':
        if args.data == 'cmnist':
            classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
            biased_classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
            nets = Munch(classifier=classifier,
                         biased_classifier=biased_classifier)
            #TODO: Cifar10 resize 32x32 with crop?
        else:
            raise NotImplementedError()
        return nets

    elif args.mode == 'featureswap':
        if args.data == 'cmnist':
            debiased_model = CNN() if not args.cmnist_use_mlp else MLP()
            biased_model = CNN() if not args.cmnist_use_mlp else MLP()

            debiased_classifier = FC(debiased_model.last_dim*2, num_classes=10)
            biased_classifier = FC(biased_model.last_dim*2, num_classes=10)

            nets = Munch(biased_F=biased_model,
                        debiased_F=debiased_model,
                        biased_C=biased_classifier,
                        debiased_C=debiased_classifier)
            return nets
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

