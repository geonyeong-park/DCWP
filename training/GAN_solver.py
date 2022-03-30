import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.checkpoint import CheckpointIO
from data.data_loader import InputFetcher
import util.utils as utils
from model.build_models import build_model
from util.params import config
from training.base_solver import BaseSolver
from training.loss import adv_loss, r1_reg
import logging


class GANSolver(BaseSolver):
    def __init__(self, args):
        super(GANSolver, self).__init__(args)
        if args.GAN_resume_iter > 0:
            self._load_checkpoint(args.GAN_resume_iter, token='GAN')

    def train_with_sup_data(self, inputs, i):
        x_sup, y, idx = inputs.x_sup, inputs.y, inputs.index
        y_trg = inputs.y_trg
        c_src, c_trg = inputs.c_src, inputs.c_trg

        # train the discriminator
        d_loss, d_losses_latent = self.compute_d_loss(self.nets, self.args, x_sup, y, y_trg, c_trg, idx)
        self._reset_grad()
        d_loss.backward()
        self.optims.biased_D.step()
        self.optims.debiased_D.step()

        # train the generator for every args.g_every steps
        if (i+1) % self.args.g_every == 0:
            g_loss, g_losses_latent = self.compute_g_loss(
                self.nets, self.args, x_sup, y, y_trg, c_src, c_trg)
            self._reset_grad()
            g_loss.backward()
            self.optims.generator.step()
            self.optims.mapping_network.step()

        return d_losses_latent, g_losses_latent

    def compute_d_loss(self, nets, args, x_sup, y, y_trg, c_trg, idx):
        x_sup.requires_grad_()

        logit_real, logit_b, logit_d, loss_weight = self.compute_logit_and_weight(idx, x_sup, y)
        loss_real = adv_loss(logit_real, 1)
        loss_reg = r1_reg(logit_real, x_sup)

        loss_b_update = self.bias_criterion(logit_b, y)
        loss_d_update = self.criterion(logit_d, y) * loss_weight.to(self.device)
        loss_b = loss_b_update.mean()
        loss_d = loss_d_update.mean()

        # with fake images
        with torch.no_grad():
            s_trg = nets.mapping_network(c_trg, y_trg)
            x_fake = nets.generator(x_sup, s_trg)

        logit_fake, _ = nets.biased_D(x_fake.detach())
        loss_fake = adv_loss(logit_fake, 0)

        loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_bias * loss_b + args.lambda_debias * loss_d

        return loss, Munch(real=loss_real.item(),
                           fake=loss_fake.item(),
                           reg=loss_reg.item(),
                           bias=loss_b.item(),
                           debias=loss_d.item())

    def compute_g_loss(self, nets, args, x_sup, y, y_trg, c_src, c_trg):
        # adversarial loss
        s_trg = nets.mapping_network(c_trg, y_trg)
        x_fake = nets.generator(x_sup, s_trg)
        logit_fake, logit_b = nets.biased_D(x_fake)
        loss_adv = adv_loss(logit_fake, 1)

        # classification loss
        loss_bias = self.criterion(logit_b, y_trg)
        loss_debias = self.criterion(logit_b, y)

        # Target-to-original domain.
        s_src = nets.mapping_network(c_src, y)
        x_recon = nets.generator(x_fake, s_src)
        loss_cyc = torch.mean(torch.abs(x_sup - x_recon))

        loss = loss_adv + args.lambda_cyc * loss_cyc \
            + args.lambda_bias * loss_bias \
            + args.lambda_debias * loss_debias

        return loss, Munch(adv=loss_adv.item(),
                           cyc=loss_cyc.item(),
                           bias=loss_bias.item(),
                           debias=loss_debias.item())

    def train_GAN(self, loaders):
        logging.info('=== Start training GAN ===')

        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        train_target_attr = []
        for data in loaders.sup_dataset.dataset.data:
            fname = os.path.relpath(data, loaders.sup_dataset.dataset.header_dir)
            train_target_attr.append(int(fname.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        self.sample_loss_ema_b = utils.EMA(train_target_attr, num_classes=self.num_classes, alpha=0.7)
        self.sample_loss_ema_d = utils.EMA(train_target_attr, num_classes=self.num_classes, alpha=0.7)

        fetcher = InputFetcher(loaders.sup,
                               loaders.unsup,
                               use_unsup_data=self.args.use_unsup_data,
                               mode='augment',
                               latent_dim=args.latent_dim,
                               num_classes=self.num_classes)

        start_time = time.time()

        for i in range(args.GAN_total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            d_losses_latent, g_losses_latent = self.train_with_sup_data(inputs, i)

            utils.moving_average_param(nets.generator, nets_ema.generator, beta=0.999)
            utils.moving_average_param(nets.mapping_network, nets_ema.mapping_network, beta=0.999)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%f]" % (elapsed, i+1, args.GAN_total_iters,
                                                                       self.scheduler.generator.get_lr()[0])
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, g_losses_latent],
                                        ['D/', 'G/']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.8f]  ' % (key, value) for key, value in all_losses.items()])
                print(log)
                logging.info(log)

            if (i+1) % args.eval_every == 0:
                print(args.data)
                utils.debug_image(nets_ema, args, inputs, i,
                                  denormalize=False if args.data == 'cmnist' else True)

            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='GAN')

