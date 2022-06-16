import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging

import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

import util.utils as utils
from data.data_loader import InputFetcher
from data.data_loader import get_original_loader, get_val_loader
from model.build_models import build_model
from training.solver import Solver
from prune.Loss import DebiasedSupConLoss


class PruneSolver(Solver):
    def __init__(self, args):
        super(PruneSolver, self).__init__(args)

        self.optims_main = Munch() # Used in retraining
        self.optims_mask = Munch() # Used in learning pruning parameter

        for net, m in self.nets.items():
            prune_param = [p for n,p in m.named_parameters() if 'gumbel_pi' in n]
            main_param = [p for n,p in m.named_parameters() if 'gumbel_pi' not in n]

            if args.optimizer == 'Adam':
                self.optims_main[net] = torch.optim.Adam(
                    params=main_param, #self.nets[net].parameters(),
                    lr=args.lr_main,
                    betas=(args.beta1, args.beta2),
                    weight_decay=0
                )
            elif args.optimizer == 'SGD':
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
        if not args.no_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler_main[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims_main[net], step_size=args.lr_decay_step_main, gamma=args.lr_gamma_main)

        self.con_criterion = DebiasedSupConLoss()

    def sparsity_regularizer(self, token='gumbel_pi'):
        reg = 0.
        for n, p in self.nets.classifier.named_parameters():
            if token in n:
                reg = reg + p.sum()
        return reg

    def save_wrong_idx(self, loader):
        self.nets.classifier.eval()
        self.nets.biased_classifier.eval()

        iterator = enumerate(loader)
        total_wrong, total_num = 0, 0
        wrong_idx = torch.empty(0).to(self.device)
        debias_idx = torch.empty(0).to(self.device)
        fname_full = []

        for _, (idx, data, attr, fname) in iterator:
            idx = idx.to(self.device)
            label = attr[:, 0].to(self.device)
            bias_label = attr[:, 1].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                if self.args.select_with_GCE:
                    logit = self.nets.biased_classifier(data)
                else:
                    logit = self.nets.classifier(data)

                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                wrong = (pred != label).long()
                debiased = (label != bias_label).long()

                total_wrong += wrong.sum()
                total_num += wrong.shape[0]
                wrong_idx = torch.cat((wrong_idx, idx[wrong == 1])).long()
                debias_idx = torch.cat((debias_idx, idx[debiased == 1])).long()

            fname_full = fname_full + fname

        assert total_wrong == len(wrong_idx)
        print('Number of wrong samples: ', total_wrong)
        self.confirm_pseudo_label(wrong_idx, debias_idx, total_num)

    def confirm_pseudo_label(self, wrong_idx, debias_idx, total_num):
        wrong_label = torch.zeros(total_num).to(self.device)
        debias_label = torch.zeros(total_num).to(self.device)

        for idx in wrong_idx:
            wrong_label[idx] = 1
        for idx in debias_idx:
            debias_label[idx] = 1

        spur_precision = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(wrong_label)
        print("Spurious precision", spur_precision)
        spur_recall = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(debias_label)
        print("Spurious recall", spur_recall)

        wrong_idx_path = ospj(self.args.checkpoint_dir, 'wrong_index.pth')

        if not self.args.supervised:
            torch.save(wrong_label, wrong_idx_path)
        else:
            torch.save(debias_label, wrong_idx_path)
        print('Saved wrong index label.')
        self.nets.classifier.train()
        self.nets.biased_classifier.train()

    def save_high_score_idx(self, loader):
        self.nets.classifier.eval()
        self.nets.biased_classifier.eval()

        iterator = enumerate(loader)
        total_num = 0
        total_idx = torch.empty(0).to(self.device)
        score_array = torch.empty(0).to(self.device)
        debias_idx = torch.empty(0).to(self.device)
        fname_full = []

        for _, (idx, data, attr, fname) in iterator:
            idx = idx.to(self.device)
            label = attr[:, 0].to(self.device)
            bias_label = attr[:, 1].to(self.device)
            data = data.to(self.device)
            debiased = (label != bias_label).long()

            with torch.no_grad():
                if self.args.select_with_GCE:
                    logit = self.nets.biased_classifier(data)
                else:
                    raise ValueError("'score' mode should use GCE biased model")

                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).float()
                max_softmax = nn.Softmax()(logit).max(dim=1).values
                score = torch.abs(correct - max_softmax)

                total_num += correct.shape[0]
                total_idx = torch.cat((total_idx, idx)).long()
                score_array = torch.cat((score_array, score))
                debias_idx = torch.cat((debias_idx, idx[debiased == 1])).long()

        mean_score = score_array.mean()
        print(mean_score)
        wrong_array = (score_array > mean_score).long()
        wrong_idx = total_idx[wrong_array == 1]
        print('Number of high score samples: ', len(wrong_idx))
        self.confirm_pseudo_label(wrong_idx, debias_idx, total_num)

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

        sampling_weight = wrong_label if not args.uniform_weight else torch.ones_like(wrong_label)
        balanced_loader = get_original_loader(args, sampling_weight=sampling_weight)

        fetcher = InputFetcher(balanced_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(True)

        for i in range(iters):
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            bias_label = torch.index_select(wrong_label, 0, idx.long())

            pred, feature = self.nets.classifier(x, feature=True)
            loss_main = self.criterion(pred, label).mean()
            loss_reg = self.sparsity_regularizer()
            loss_con = self.con_criterion(F.normalize(feature, dim=1).unsqueeze(1), label, bias_label)
            loss = loss_main + args.lambda_sparse * loss_reg + args.lambda_con_prune * loss_con

            self._reset_grad()
            loss.backward()
            optims.classifier.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f], "\
                    "Loss_main [%.6f] Loss_reg [%.6f] Loss_con [%.6f]" % (elapsed, i+1, iters,
                                                                          optims.classifier.param_groups[-1]['lr'],
                                                                          loss_main.item(),
                                                                          loss_reg.item(),
                                                                          loss_con.item())
                print(log)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='prune')

            if (i+1) % args.eval_every == 0:
                total = 0
                active = 0
                layerwise = {}
                for n, p in self.nets.classifier.named_parameters():
                    if 'gumbel_pi' in n:
                        active_n = (p>=0).sum().item()
                        total_n = torch.ones_like(p).sum().detach().item()
                        layerwise[n] = active_n / total_n

                        total += total_n
                        active += active_n
                        if active_n == 0: raise ValueError('Warning: Dead layer')

                ratio = active / total
                print('ratio:', ratio)
                self.valid_logger.append(ratio, which='ratio')
                self.valid_logger.append(layerwise, which='layerwise_ratio')

                self.nets.classifier.pruning_switch(False)
                self.nets.classifier.freeze_switch(True)
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='prune')
                self.nets.classifier.pruning_switch(True)
                self.nets.classifier.freeze_switch(False)

    def retrain(self, iters, freeze=True):
        args = self.args
        nets = self.nets
        optims = self.optims_main # Train only weight parameter

        wrong_label = torch.load(ospj(self.args.checkpoint_dir, 'wrong_index.pth'))
        print('Number of wrong samples: ', wrong_label.sum())
        upweight = torch.ones_like(wrong_label)
        upweight[wrong_label == 1] = args.lambda_upweight

        upweight_loader = get_original_loader(args, sampling_weight=upweight)

        fetcher = InputFetcher(upweight_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(freeze)

        for i in range(iters):
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            bias_label = torch.index_select(wrong_label, 0, idx.long())

            pred, feature = self.nets.classifier(x, feature=True)
            loss_main = self.criterion(pred, label).mean()  #TODO: loss_con
            loss_con = self.con_criterion(F.normalize(feature, dim=1).unsqueeze(1), label, bias_label)
            loss = loss_main + args.lambda_con_retrain * loss_con

            self._reset_grad()
            loss.backward()
            optims.classifier.step()

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f], "\
                    "Loss_main [%.6f] Loss_con [%.6f] " % (elapsed, i+1, iters,
                                                           optims.classifier.param_groups[-1]['lr'],
                                                           loss_main.item(),
                                                           loss_con.item())
                print(log)

            # save model checkpoints
            if (i+1) % args.save_every_retrain == 0:
                self._save_checkpoint(step=i+1, token='retrain')

            if (i+1) % args.eval_every_retrain == 0:
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='retrain')
                self.valid_logger.append(total_acc.item(), which='retrain')

            if not self.args.no_lr_scheduling:
                self.scheduler_main.classifier.step()

    def train(self):
        logging.info('=== Start training ===')
        """
        0. Pretrain model. Save pretrained ckpt
        1. Load pretrained model and pseudo bias label
        2. Build balanced dataset. Train pruning parameters
        3. Resume training with learned pruning parameters
        """

        args = self.args
        loader = self.loaders.train

        try:
            self._load_checkpoint(args.pretrain_iter, 'pretrain')
            print('Pretrained ckpt exists. Checking upweight index ckpt...')
        except:
            print('Start pretraining...')
            self.train_ERM(args.pretrain_iter)
            self._load_checkpoint(args.pretrain_iter, 'pretrain')

        if os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth')):
            print('Upweight ckpt exists.')
        else:
            print('Upweight ckpt does not exist. Creating...')
            if args.pseudo_label_method == 'wrong':
                self.save_wrong_idx(loader)
            elif args.pseudo_label_method == 'score':
                self.save_high_score_idx(loader)

        assert os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth'))

        try:
            self._load_checkpoint(args.pruning_iter, 'prune')
            print('Pruning parameter ckpt exists. Start retraining...')
        except:
            print('Pruning parameter ckpt does not exist. Start pruning...')
            self.train_PRUNE(args.pruning_iter)
        #TODO: Failed to reproduce JTT. Run original implementations of JTT
        self.valid_logger.save()

        if self.args.reinitialize:
            # NOT USED. Reinitialization performs worse
            reinit_dict = torch.load(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'.format(0, 'initial')))['classifier']
            mask_dict = torch.load(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'.format(args.pruning_iter, 'prune')))['classifier']
            pruning_dict = {k: v for k, v in mask_dict.items() if 'gumbel_pi' in k}
            reinit_dict.update(pruning_dict) # Update pruning parameters only
            self.nets.classifier.load_state_dict(reinit_dict)
            print('Reinitialized model from ', ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'.format(0, 'initial')))

        self.retrain(args.retrain_iter, freeze=True)
        self.valid_logger.save()
        print('Finished training')

    def evaluate(self):
        fetcher_val = self.loaders.val
        self._load_checkpoint(self.args.retrain_iter, 'retrain')
        print('Load model from ', ospj(self.args.checkpoint_dir, '{:06d}_{}_nets.ckpt'.format(self.args.retrain_iter, 'retrain')))
        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(False)

        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0, which='Test', save_in_result=True)

        self._tsne(fetcher_val)

