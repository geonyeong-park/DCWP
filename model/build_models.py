from munch import Munch
import copy
import numpy as np
from model.generator import ResGenerator, MLPDiscriminator, ConvDiscriminator, MappingNetwork
from model.mlp import MLP
from util.params import config


def build_model(args):
    if args.mode == 'augment':
        return build_augment(args)
    else:
        return build_debias(args)


def build_augment(args):
    # Generative model for augmentation, biased model, GAN discriminator (if need)
    if args.data == 'cmnist':
        debiased_discriminator = MLPDiscriminator()
        biased_discriminator = MLPDiscriminator()
        generator = ResGenerator(img_size=config[args.data]['size'],
                                 num_channels=config[args.data]['in_channels'],
                                 dim_in=args.dim_in,
                                 style_dim=args.style_dim,
                                 sigmoid_output=True)
        mapping_network = MappingNetwork(args.latent_dim, args.style_dim,
                                         num_domains=config[args.data]['num_classes'])

        generator_ema = copy.deepcopy(generator)
        mapping_network_ema = copy.deepcopy(mapping_network)

        nets = Munch(generator=generator,
                     mapping_network=mapping_network,
                     biased_D=biased_discriminator,
                     debiased_D=debiased_discriminator)
        nets_ema = Munch(generator=generator_ema,
                        mapping_network=mapping_network_ema)

        return nets, nets_ema

    else:
        pass


def build_debias(args):
    # Main model only
    if args.data == 'cmnist':
        debiased_model = MLP()
    else:
        pass

    nets = Munch(biased_model=debiased_model)

    return nets
