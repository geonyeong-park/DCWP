import torch
from munch import Munch
from model.simple_model import CNN, MLP, FC
from model.resnet import ResNet18, ResNet34
from model.wide_resnet import WideResNet28_10, WideResNet16_8

from prune.GateSimpleModel import GateCNN, GateFCN
from prune.GateResnet import GateResNet18, GateResNet34
from prune.GateWideResnet import GateWideResNet28_10, GateWideResNet16_8

from data.transforms import num_classes

def build_model(args):
    n_classes = num_classes[args.data]
    if args.mode in ['prune', 'JTT', 'MRM', 'ERM']: # ERM is included for coding consistency. pruning X
        if args.data == 'cmnist':
            classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
            biased_classifier = GateCNN() if not args.cmnist_use_mlp else GateFCN()
            nets = Munch(classifier=classifier,
                         biased_classifier=biased_classifier)
        else:
            classifier = GateResNet18(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
            biased_classifier = GateResNet18(IMAGENET_pretrained=args.imagenet, n_classes=n_classes)
            nets = Munch(classifier=classifier,
                         biased_classifier=biased_classifier)
        return nets

    elif args.mode == 'featureswap':
        if args.data == 'cmnist':
            # Exactly same as GateModel except pruning parameters
            debiased_model = CNN() if not args.cmnist_use_mlp else MLP()
            biased_model = CNN() if not args.cmnist_use_mlp else MLP()

            debiased_classifier = FC(debiased_model.last_dim*2, num_classes=10)
            biased_classifier = FC(biased_model.last_dim*2, num_classes=10)

            nets = Munch(biased_F=biased_model,
                        debiased_F=debiased_model,
                        biased_C=biased_classifier,
                        debiased_C=debiased_classifier)
        else:
            debiased_model = ResNet18(IMAGENET_pretrained=args.imagenet)
            biased_model = ResNet18(IMAGENET_pretrained=args.imagenet)

            debiased_classifier = FC(debiased_model.last_dim*2, num_classes=n_classes)
            biased_classifier = FC(biased_model.last_dim*2, num_classes=n_classes)

            nets = Munch(biased_F=biased_model,
                        debiased_F=debiased_model,
                        biased_C=biased_classifier,
                        debiased_C=debiased_classifier)

        return nets
    else:
        raise NotImplementedError()

