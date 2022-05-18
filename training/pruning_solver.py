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
from torch.utils import data

from util.checkpoint import CheckpointIO
from util.params import config
import util.utils as utils
from util.utils import MultiDimAverageMeter
from data.data_loader import InputFetcher
from model.build_models import build_model
from training.loss import GeneralizedCELoss
from training.solver import Solver
from data.data_loader import get_original_loader, get_val_loader


class PruneSolver(Solver):
    def __init__(self, args):
        super(PruneSolver, self).__init__(args)

        self.optims_main = Munch() # Used in retraining
        self.optims_mask = Munch() # Used in learning pruning parameter

        for net, m in self.nets.items():
            prune_param = [p for n,p in m.named_parameters() if 'gumbel_pi' in n]
            main_param = [p for n,p in m.named_parameters() if 'gumbel_pi' not in n]
            self.optims_main[net] = torch.optim.SGD(
                main_param,
                lr=args.lr_main,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
            self.optims_mask[net] = torch.optim.Adam(
                prune_param,
                lr=args.lr_prune,
            )

        self.scheduler_main = Munch() # Used in retraining
        if args.do_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler_main[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims[net], step_size=args.lr_decay_step, gamma=args.lr_gamma)

    def sparsity_regularizer(self, token='gumbel_pi'):
        reg = 0.
        for n, p in self.nets.classifier.named_parameters():
            if token in n:
                reg = reg + p.sum()
        return reg

    def save_wrong_idx(self, loader):
        self.nets.classifier.eval()
        iterator = enumerate(loader)
        total_wrong, total_num = 0, 0
        wrong_idx = torch.empty(0).to(self.device)

        for _, (idx, data, attr, fname) in iterator:
            idx = idx.to(self.device)
            label = attr[:, 0].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                logit = self.nets.classifier(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                wrong = (pred != label).long()

                total_wrong += wrong.sum()
                total_num += wrong.shape[0]
                wrong_idx = torch.cat((wrong_idx, idx[wrong == 1])).long()
        assert total_wrong == len(wrong_idx)
        print('Number of wrong samples: ', total_wrong)
        wrong_label = torch.zeros(total_num).to(self.device)
        for idx in wrong_idx:
            wrong_label[idx] = 1

        wrong_idx_path = ospj(self.args.checkpoint_dir, 'wrong_index.pth')
        torch.save(wrong_label, wrong_idx_path)
        print('Saved wrong index label.')
        self.nets.classifier.train()

    def train_PRUNE(self, iters):
        args = self.args
        nets = self.nets
        optims = self.optims_mask # Train only pruning parameter

        # Load and balance data
        wrong_label = torch.load(ospj(self.args.checkpoint_dir, 'wrong_index.pth'))
        remain = 1. - wrong_label
        subsampled_idx = remain.multinomial(min(int(wrong_label.sum() / self.attr_dims[1]), 1)).long()
        for idx in subsampled_idx:
            wrong_label[idx] = 1

        balanced_loader = get_original_loader(args, sampling_weight=wrong_label)

        fetcher = InputFetcher(balanced_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(True)

        for i in range(iters):
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            pred = self.nets.classifier(x)
            loss_main = self.criterion(pred, label).mean()
            loss_reg = self.sparsity_regularizer()
            loss = loss_main + args.lambda_sparse * loss_reg

            self._reset_grad()
            loss.backward()
            optims.classifier.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f], "\
                    "Loss_main [%.6f] Loss_reg [%.6f]" % (elapsed, i+1, iters,
                                                          optims.classifier.param_groups[-1]['lr'],
                                                          loss_main.item(),
                                                          loss_reg.item())
                print(log)

                total = 0
                active = 0
                for n, p in self.nets.classifier.named_parameters():
                    if 'gumbel_pi' in n:
                        active_n = (p>=0).sum().item()
                        total_n = torch.ones_like(p).sum().detach().item()

                        total += total_n
                        active += active_n
                        if active_n==0: print(n)
                print('ratio:', active/total)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='prune')

            if (i+1) % args.eval_every == 0:
                self.nets.classifier.pruning_switch(False)
                self.nets.classifier.freeze_switch(True)
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='prune')
                self.nets.classifier.pruning_switch(True)
                self.nets.classifier.freeze_switch(False)

    def retrain(self, iters):
        args = self.args
        nets = self.nets
        optims = self.optims_main # Train only pruning parameter

        wrong_label = torch.load(ospj(self.args.checkpoint_dir, 'wrong_index.pth'))
        upweight = torch.ones_like(wrong_label)
        upweight[wrong_label == 1] = args.lambda_upweight

        upweight_loader = get_original_loader(args, sampling_weight=upweight)

        fetcher = InputFetcher(upweight_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(True)

        for i in range(iters):
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            pred = self.nets.classifier(x)
            loss = self.criterion(pred, label).mean()  #TODO: loss_con

            self._reset_grad()
            loss.backward()
            optims.classifier.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f], "\
                    "Loss_main [%.6f] " % (elapsed, i+1, iters, optims.classifier.param_groups[-1]['lr'], loss.item())
                print(log)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='retrain')

            if (i+1) % args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='retrain')

            if self.args.do_lr_scheduling:
                self.scheduler_main.classifier.step()

    def train(self):
        logging.info('=== Start training ===')
        """
        0. (optional) Pretrain model. Save initial and pretrained ckpt
        1. Load pretrained model. Select wrong data
        2. Build balanced dataset. Train pruning parameters
        3. Reload initial ckpt and apply pruning
        4. Retrain with wrong data
        """

        args = self.args
        loader = self.loaders.train

        try:
            self._load_checkpoint(args.pretrain_iter, 'pretrain')
            print('Pretrained ckpt exists. Checking upweight index ckpt...')
            assert os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth')), \
                print('Upweight ckpt does not exist.')
            print('Upweight ckpt exists.')
        except:
            print('Start pretraining...')
            self.train_ERM(args.pretrain_iter)
            self._load_checkpoint(args.pretrain_iter, 'pretrain')
            self.save_wrong_idx(loader)

        assert os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth'))
        try:
            self._load_checkpoint(args.pruning_iter, 'prune')
            print('Pruning parameter ckpt exists. Start retraining...')
        except:
            print('Pruning parameter ckpt does not exist. Start pruning...')
            self.train_PRUNE(args.pruning_iter)

        self.retrain(args.retrain_iter)

        """
        self._load_checkpoint(0, 'initialization')

        torch.load(
        pruning_dict = {k: v for k, v in mask.state_dict().items() if 'gumbel_pi' in k}

        model_reinit = get_model(   # Reload IMAGENET pretrained weights again
            model=args.model,
            pretrained=not args.train_from_scratch,
            resume=resume,
            n_classes=train_data.n_classes,
            dataset=args.dataset,
            log_dir=args.log_dir,
            prune=args.prune,
            biased_ckpt=None
        )
        reinit_dict = model_reinit.state_dict()
        reinit_dict.update(pruning_dict) # Update pruning parameters only

        self._load_checkpoint(0, 'initialization')
        self.retrain(args.retrain_iter)
        print('Finished training')
        """





