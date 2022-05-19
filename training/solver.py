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

from data.data_loader import get_original_loader, get_val_loader


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config[args.data]['num_classes']
        self.attr_dims = [self.num_classes, self.num_classes]

        self.nets = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)

        self.optims = Munch() # Used in pretraining
        for net in self.nets.keys():
            self.optims[net] = torch.optim.SGD(
                self.nets[net].parameters(),
                lr=args.lr_pre,
                momentum=0.9,
                weight_decay=args.weight_decay
            )

        self.scheduler = Munch()
        if args.do_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims[net], step_size=args.lr_decay_step_pre, gamma=args.lr_gamma_pre)

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'), **self.nets),
        ]
        logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'),
                            level=logging.INFO)

        self.to(self.device)
        self.bias_criterion = GeneralizedCELoss()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # BUILD LOADERS
        self.loaders = Munch(train=get_original_loader(args),
                             val=get_val_loader(args))

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

    def _load_checkpoint(self, step, token, which=None, return_fname=False):
        for ckptio in self.ckptios:
            ckptio.load(step, token, which, return_fname)

    def validation(self, fetcher):
        self.nets.classifier.eval()

        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)
        iterator = enumerate(fetcher)

        total_correct, total_num = 0, 0

        for index, (_, data, attr, fname) in iterator:
            label = attr[:, 0].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                logit = self.nets.classifier(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

                total_correct += correct.sum()
                total_num += correct.shape[0]

            attr = attr[:, [0, 1]]
            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        total_acc = total_correct / float(total_num)
        accs = attrwise_acc_meter.get_mean()

        self.nets.classifier.train()
        return total_acc, accs

    def report_validation(self, valid_attrwise_acc, valid_acc, step, which='bias'):
        eye_tsr = torch.eye(self.attr_dims[0]).long()
        valid_acc_align = valid_attrwise_acc[eye_tsr == 1].mean().item()
        valid_acc_conflict = valid_attrwise_acc[eye_tsr == 0].mean().item()

        all_acc = dict()
        for acc, key in zip([valid_acc, valid_acc_align, valid_acc_conflict],
                                ['Acc/total', 'Acc/align', 'Acc/conflict']):
            all_acc[key] = acc
        log = f"({which} Validation) Iteration [{step+1}], "
        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_acc.items()])
        print(log)
        logging.info(log)

    def train_ERM(self, iters):
        logging.info('=== Start training ===')
        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(self.loaders.train)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self._save_checkpoint(step=0, token='initial')

        for i in range(iters):
            # fetch images and labels
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            pred = self.nets.classifier(x)
            pred_bias = self.nets.biased_classifier(x)

            loss = self.criterion(pred, label).mean()
            loss_bias = self.bias_criterion(pred_bias, label).mean()

            self._reset_grad()
            loss.backward()
            loss_bias.backward()
            optims.classifier.step()
            optims.biased_classifier.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f]" % (elapsed, i+1, iters,
                                                                           optims.classifier.param_groups[-1]['lr'])

                all_losses = dict()
                for loss, key in zip([loss.item(), loss_bias.item()], ['Debiased/',
                                                                       'Biased/']):
                    all_losses[key] = loss
                log += ' '.join(['%s: [%f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                logging.info(log)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='pretrain')

            if (i+1) % args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='main')

            if self.args.do_lr_scheduling:
                self.scheduler.classifier.step()
                self.scheduler.biased_classifier.step()
