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
from training.vae_solver import VAESolver


class AugmentSolver(VAESolver):
    """
    1. train biased model
    2. train vae
    3. L2-attack vae model
    4. Generate and save final adversary
    """
    def __init__(self, args):
        super(AugmentSolver, self).__init__(args)
        if args.vae_resume_iter > 0:
            self._load_checkpoint(args.vae_resume_iter, 'vae')

    def generate_adv_mu(self, x, attr):
        mu = self.nets.generator(x)[2].clone().detach()
        adv_mu = mu.clone().detach()
        batch_size = len(x)
        y = attr[:, 0]

        ori_norm = torch.norm(mu, p=2, dim=1)

        for i in range(self.args.attack_iters):
            adv_mu.requires_grad = True
            x_adv = self.nets.generator.sample(adv_mu)

            pred = self.nets.biased_model(x_adv)
            cost = -F.cross_entropy(pred, y)

            grad = torch.autograd.grad(cost, adv_mu,
                                       retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-7
            grad = grad / grad_norms.view(batch_size, 1)
            adv_mu = adv_mu.detach() - 0.5*ori_norm.view(batch_size, 1) * grad

            delta = adv_mu - mu
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = 1.2 * ori_norm / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1)

            adv_mu = (mu + delta).detach()
        return adv_mu

    def augment(self, loaders):
        logging.info('=== Start augmentation ===')
        if not self.args.vae_resume_iter > 0:
            self.train_vae(loaders)
            logging.info('Successfully load pretrained VAE and biased model')

        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.sup, loaders.unsup,
                               use_unsup_data=args.use_unsup_data,
                               mode='augment')

        # for i in range(args.total_iters):
        inputs = next(fetcher)
        x_sup, attr_sup = inputs.x_sup, inputs.attr_sup

        adv_mu = self.generate_adv_mu(x_sup, attr_sup)
        x_aug = self.nets.generator.sample(adv_mu)

        fname = ospj(self.args.result_dir, '{}_aug.png'.format(1))
        utils.save_image(x_aug[:32], ncol=4, filename=fname)

        fname = ospj(self.args.result_dir, '{}_ori.png'.format(1))
        utils.save_image(x_sup[:32], ncol=4, filename=fname)







