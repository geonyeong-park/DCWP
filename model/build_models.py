from munch import Munch
import copy
import numpy as np
from model.mlp import MLP, FC, SimCLRHead
from model.generator import ResGenerator, Discriminator
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
        discriminator = Discriminator(img_size=config[args.data]['size'],
                                      num_channels=config[args.data]['in_channels'],
                                      num_classes=10,
                                      channel_base=config[args.data]['channel_base_D'],
                                      cmap_dim=config[args.data]['style_dim'],
                                      class_embed_dim=config[args.data]['class_embed_dim'],
                                      channel_max=config[args.data]['max_conv_dim'])

        generator = ResGenerator(img_size=config[args.data]['size'],
                                 num_channels=config[args.data]['in_channels'],
                                 num_classes=10,
                                 channel_base=config[args.data]['channel_base_G'],
                                 style_dim=config[args.data]['style_dim'],
                                 mapping_latent_dim=config[args.data]['mapping_latent_dim'],
                                 class_embed_dim=config[args.data]['class_embed_dim'],
                                 max_conv_dim=config[args.data]['max_conv_dim'],
                                 sigmoid_output=True)

        generator_ema = copy.deepcopy(generator)

        nets = Munch(generator=generator,
                     discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema)
        return nets, nets_ema

    else:
        pass


def build_debias(args):
    # Main model only
    if args.data == 'cmnist':
        debiased_model = MLP()
        biased_model = MLP()
        debiased_classifier = FC(MLP.last_dim, num_classes=10)
        biased_classifier = FC(MLP.last_dim, num_classes=10)

        nets = Munch(biased_F=biased_model,
                     debiased_F=debiased_model,
                     biased_C=biased_classifier,
                     debiased_C=debiased_classifier)
        return nets

    else:
        pass

