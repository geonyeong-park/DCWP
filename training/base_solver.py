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

from util.checkpoint import CheckpointIO
from util.params import config
import util.utils as utils
from util.utils import MultiDimAverageMeter
from data.data_loader import InputFetcher
from model.build_models import build_model
from training.loss import GeneralizedCELoss


class BaseSolver(nn.Module):
    def __init__(self, args):
        # build models, train biased model

        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config[args.data]['num_classes']
        self.attr_dims = [self.num_classes, self.num_classes]

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name+'_ema', module)

        self.optims = Munch()
        for net in self.nets.keys():
            if net in ['generator', 'mapping_network']:
                lr = args.lr_G
            else:
                lr = args.lr_D
            self.optims[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=lr,
                betas=(args.beta1, args.beta2)
            )

        # LR decaying: not used when training GAN
        self.scheduler = Munch()
        for net in self.nets.keys():
            self.scheduler[net] = torch.optim.lr_scheduler.StepLR(
                self.optims[net], step_size=args.lr_decay_step, gamma=args.lr_gamma)

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'), **self.nets),
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets_ema.ckpt'), **self.nets_ema)
        ]
        logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'),
                            level=logging.INFO)

        self.to(self.device)
        self.bias_criterion = GeneralizedCELoss()
        self.criterion = nn.CrossEntropyLoss()

    def _reset_grad(self):
        def _recursive_reset(optims_dict):
            for _, optim in optims_dict.items():
                if isinstance(optim, dict):
                    _recursive_reset(optim)
                else:
                    optim.zero_grad()
        return _recursive_reset(self.optims)

    def _save_checkpoint(self, step, token):
        for ckptio in self.ckptios:
            ckptio.save(step, token)

    def _load_checkpoint(self, step, token, which=None):
        for ckptio in self.ckptios:
            ckptio.load(step, token, which)

    def validation_D(self, fetcher):
        self.nets.biased_D.eval()
        self.nets.debiased_D.eval()

        attrwise_acc_meter_bias = MultiDimAverageMeter(self.attr_dims)
        attrwise_acc_meter_debias = MultiDimAverageMeter(self.attr_dims)
        iterator = enumerate(fetcher)

        for index, (_, data, attr, fname) in iterator:
            label = attr[:, 0].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                _, logit = self.nets.biased_D(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

                logit_d = self.nets.debiased_D(data)
                pred_d = logit_d.data.max(1, keepdim=True)[1].squeeze(1)
                correct_d = (pred_d == label).long()

            attr = attr[:, [0, 1]]
            attrwise_acc_meter_bias.add(correct.cpu(), attr.cpu())
            attrwise_acc_meter_debias.add(correct_d.cpu(), attr.cpu())

        accs_b = attrwise_acc_meter_bias.get_mean()
        accs_d = attrwise_acc_meter_debias.get_mean()

        self.nets.biased_D.train()
        self.nets.debiased_D.train()
        return accs_b, accs_d

    def compute_biased_loss(self, x, label):
        _, pred = self.nets.biased_D(x)
        loss_GCE = self.bias_criterion(pred, label).mean()
        return loss_GCE

    def compute_logit_and_weight(self, index, x, label):
        logit_real, logit_b = self.nets.biased_D(x)
        logit_d = self.nets.debiased_D(x)

        loss_b = self.criterion(logit_b, label).cpu().detach()
        loss_d = self.criterion(logit_d, label).cpu().detach()

        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d

        # EMA sample loss
        self.sample_loss_ema_b.update(loss_b, index)
        self.sample_loss_ema_d.update(loss_d, index)

        # class-wise normalize
        loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

        label_cpu = label.cpu()

        for c in range(self.num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = self.sample_loss_ema_b.max_loss(c)
            max_loss_d = self.sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d

        # re-weighting based on loss value / generalized CE for biased model
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')

        return logit_real, logit_b, logit_d, loss_weight

    def train_biased_model(self, loaders):
        """ NOT used """
        logging.info('=== Start training biased model ===')
        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(loaders.sup, loaders.unsup,
                               use_unsup_data=args.use_unsup_data,
                               mode='augment')
        fetcher_val = loaders.val
        start_time = time.time()

        for i in range(args.bias_total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_sup, label = inputs.x_sup, inputs.y

            loss_GCE = self.compute_biased_loss(x_sup, label)
            optims.biased_model.zero_grad()
            loss_GCE.backward()
            optims.biased_model.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f]" % (elapsed, i+1, args.bias_total_iters,
                                                                           self.scheduler.biased_model.get_lr()[0])
                all_losses = dict()
                for loss, key in zip([loss_GCE.item()], ['GCE/']):
                    all_losses[key] = loss
                log += ' '.join(['%s: [%.8f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                logging.info(log)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='bias')

            if (i+1) % args.eval_every == 0:
                valid_attrwise_acc = self.validation_D(fetcher_val)
                valid_acc = torch.mean(valid_attrwise_acc).item()

                eye_tsr = torch.eye(self.attr_dims[0]).long()
                valid_acc_align = valid_attrwise_acc[eye_tsr == 1].mean().item()
                valid_acc_conflict = valid_attrwise_acc[eye_tsr == 0].mean().item()

                all_acc = dict()
                for acc, key in zip([valid_acc, valid_acc_align, valid_acc_conflict],
                                     ['Acc/total', 'Acc/align', 'Acc/conflict']):
                    all_acc[key] = acc
                log = "(Validation) Iteration [%i/%i], " % (i+1, args.bias_total_iters)
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_acc.items()])
                print(log)
                logging.info(log)

            self.scheduler.biased_model.step()

