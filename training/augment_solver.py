import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.checkpoint import CheckpointIO
from data.data_loader import InputFetcher
import util.utils as utils
from model.build_models import build_model
from util.params import config
from training.base_solver import BaseSolver
from training.GAN_solver import GANSolver


class AugmentSolver(GANSolver):
    def __init__(self, args):
        super(AugmentSolver, self).__init__(args)
        if args.GAN_resume_iter > 0:
            self._load_checkpoint(args.GAN_resume_iter, 'GAN')

    def augment(self, loaders):
        logging.info('=== Start augmentation ===')
        if not self.args.GAN_resume_iter > 0:
            self.train_GAN(loaders)
        logging.info('Successfully load pretrained GAN and biased model')

        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.sup, loaders.unsup,
                               use_unsup_data=args.use_unsup_data,
                               mode='augment')

        inputs = next(fetcher)
        #TODO: augment every training images
        return
