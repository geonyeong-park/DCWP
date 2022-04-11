import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.utils as utils
from util.utils import MultiDimAverageMeter
from data.data_loader import InputFetcher
from training.debias_solver import DebiasSolver


class ContraSolver(DebiasSolver):
    def __init__(self, args):
        super(ContraSolver, self).__init__(args)

    def normalize(self, x, dim=1, eps=1e-8):
        return x / (x.norm(dim=dim, keepdim=True) + eps)

    def simclr_matrix(self, z1, z2):
        simclr_z1 = self.nets.head(z1)
        simclr_z2 = self.nets.head(z2)
        simclr_z1 = self.normalize(simclr_z1)
        simclr_z2 = self.normalize(simclr_z2)

        outputs = torch.cat([simclr_z1, simclr_z2], dim=0)
        sim_matrix = torch.mm(outputs, outputs.t()).to(self.device)
        return sim_matrix

    def NT_xent(self, sim_matrix, temperature=0.5, chunk=2, eps=1e-8, loss_weight=None):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        device = sim_matrix.device

        B = sim_matrix.size(0) // chunk  # B = B' / chunk

        eye = torch.eye(B * chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

        if loss_weight is not None:
            loss = torch.sum(sim_matrix[:B, B:].diag()*loss_weight + \
                             sim_matrix[B:, :B].diag()*loss_weight) / (2 * B)
        else:
            loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

        return loss

    def compute_contra_loss(self, z_b, z_l, z_b_swap=None, loss_weight=None):
        if z_b_swap is None:
            indices = np.random.permutation(z_b.size(0))
            z_b_swap = z_b[indices]         # z tilde
        indices2 = np.random.permutation(z_b.size(0))
        z_b_swap2 = z_b[indices2]

        z_conflict_1 = torch.cat((z_l, z_b_swap), dim=1)
        z_conflict_2 = torch.cat((z_l, z_b_swap2), dim=1)

        mat = self.simclr_matrix(z_conflict_1, z_conflict_2)
        loss_contra = self.NT_xent(mat, loss_weight=None)
        return loss_contra

    def compute_unsup_feature(x_unsup):
        pass

    def train(self, loaders):
        logging.info('=== Start training ===')
        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.sup, loaders.unsup,
                               use_unsup_data=args.use_unsup_data,
                               mode='ours')
        fetcher_val = loaders.val
        start_time = time.time()
        self.set_loss_ema(loaders)

        for i in range(args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            idx, x_sup, label, fname = inputs.index, inputs.x_sup, inputs.y, inputs.fname_sup
            z_l, z_b, loss_dis_conflict, loss_dis_align, loss_weight = self.compute_dis_loss(x_sup, idx, label)

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if i+1 > args.swap_iter:
                z_b_swap, label_swap, loss_swap_conflict, loss_swap_align = self.compute_swap_loss(z_b, z_l, label, loss_weight)
                loss_contra = self.compute_contra_loss(z_b, z_l, z_b_swap, loss_weight)
            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float().to(self.device)
                loss_swap_align = torch.tensor([0]).float().to(self.device)
                loss_contra = torch.tensor([0]).float().to(self.device)

            loss_dis  = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()
            loss = loss_dis + self.args.lambda_swap * loss_swap + self.args.lambda_contra * loss_contra

            self._reset_grad()
            loss.backward()
            optims.biased_F.step()
            optims.debiased_F.step()
            optims.biased_C.step()
            optims.debiased_C.step()
            optims.head.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f]" % (elapsed, i+1, args.total_iters,
                                                                           optims.biased_F.param_groups[-1]['lr'])

                all_losses = dict()
                for loss, key in zip([loss_dis_conflict.mean().item(), loss_dis_align.mean().item(),
                                      loss_swap_conflict.mean().item(), loss_swap_align.mean().item(),
                                      loss_contra.item()],
                                     ['dis_conflict/', 'dis_align/',
                                      'swap_conflict/', 'swap_align/', 'Contra/']):
                    all_losses[key] = loss
                log += ' '.join(['%s: [%f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                print('Average Loss weight:', loss_weight.mean().item())
                logging.info(log)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='debias')

            if (i+1) % args.eval_every == 0:
                total_acc_b, total_acc_d, valid_attrwise_acc_b, valid_attrwise_acc_d = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc_b, total_acc_b, i, which='bias')
                self.report_validation(valid_attrwise_acc_d, total_acc_d, i, which='debias')

            if i+1 >= args.swap_iter:
                self.scheduler.biased_F.step()
                self.scheduler.debiased_F.step()
                self.scheduler.biased_C.step()
                self.scheduler.debiased_C.step()
                self.scheduler.head.step()

