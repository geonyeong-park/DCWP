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
import logging


class VAESolver(BaseSolver):
    def __init__(self, args):
        super(VAESolver, self).__init__(args)
        if args.bias_resume_iter > 0:
            self._load_checkpoint(args.bias_resume_iter, token='bias', which='biased_model')

    def vae_loss(self, x):
        results = self.nets.generator.forward(x)
        train_loss_dict = self.nets.generator.loss_function(*results)
        return train_loss_dict

    def train_vae(self, loaders):
        logging.info('=== Start training VAE ===')
        if not self.args.bias_resume_iter > 0:
            self.train_biased_model(loaders)
            logging.info('Successfully load pretrained biased model')

        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.concat, # We concatenate labeled+unlabeled dataset
                               use_unsup_data=False, # Only for syntax. We indeed use unsup data in VAE
                               mode='augment')

        start_time = time.time()

        for i in range(args.vae_total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_sup, attr_sup = inputs.x_sup, inputs.attr_sup # x_sup includes both labeled and unlabeled one only in this Solver.

            train_loss_dict = self.vae_loss(x_sup)
            loss_vae = train_loss_dict['loss']

            self._reset_grad()
            loss_vae.backward()
            optims.generator.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%f]" % (elapsed, i+1, args.vae_total_iters,
                                                                       self.scheduler.generator.get_lr()[0])
                all_losses = dict()
                for key, loss in train_loss_dict.items():
                    all_losses[key] = loss
                log += ' '.join(['%s: [%.8f]  ' % (key, value) for key, value in all_losses.items()])
                print(log)
                logging.info(log)

            if (i+1) % args.eval_every == 0:
                x_recon = self.nets.generator.generate(x_sup)
                fname = ospj(self.args.result_dir, '{}_vae_recon.png'.format(i+1))
                utils.save_image(x_recon[:32], ncol=4, filename=fname)

            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='vae')

            self.scheduler.generator.step()
