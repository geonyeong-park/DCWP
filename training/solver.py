import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging
import pickle as pkl

import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from util.checkpoint import CheckpointIO
import util.utils as utils
from data.transforms import num_classes

from util.utils import MultiDimAverageMeter, ValidLogger
from data.data_loader import InputFetcher
from model.build_models import build_model
from training.loss import GeneralizedCELoss

from data.data_loader import get_original_loader, get_val_loader


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes[args.data]
        self.attr_dims = [self.num_classes, self.num_classes]

        self.nets = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)

        self.optims = Munch() # Used in pretraining
        for net in self.nets.keys():
            if args.optimizer == 'Adam':
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr_pre,
                    betas=(args.beta1, args.beta2),
                    weight_decay=0
                )
            elif args.optimizer == 'SGD':
                self.optims[net] = torch.optim.SGD(
                    self.nets[net].parameters(),
                    lr=args.lr_pre,
                    momentum=0.9,
                    weight_decay=args.weight_decay
                )

        self.scheduler = Munch()
        if not args.no_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims[net], step_size=args.lr_decay_step_pre, gamma=args.lr_gamma_pre)

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'), **self.nets),
        ]
        logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'),
                            level=logging.INFO)


        self.valid_logger = ValidLogger(ospj(args.log_dir, f'valid_acc_{args.pruning_iter}.pkl'))


        self.tsne = TSNE(n_components=2, perplexity=20, init='pca', n_iter=3000)

        self.to(self.device)
        self.bias_criterion = GeneralizedCELoss()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # BUILD LOADERS
        self.loaders = Munch(train=get_original_loader(args),
                             val=get_val_loader(args),
                             trainset=get_original_loader(args, return_dataset=True))

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

    def update_pseudo_label(self, bias_score_array, loader, iters, pseudo_every):
        self.nets.biased_classifier.eval()

        iterator = enumerate(loader)
        debias_idx = torch.empty(0).to(self.device)

        for _, (idx, data, attr, fname) in iterator:
            idx = idx.to(self.device)
            label = attr[:, 0].to(self.device)
            bias_label = attr[:, 1].to(self.device)
            data = data.to(self.device)
            debiased = (label != bias_label).long()

            with torch.no_grad():
                logit = self.nets.biased_classifier(data)
                bias_prob = nn.Softmax()(logit)[torch.arange(logit.size(0)), label]
                bias_score = 1 - bias_prob

                for i, v in enumerate(idx):
                    bias_score_array[v] += bias_score[i] * pseudo_every / iters

                debias_idx = torch.cat((debias_idx, idx[debiased == 1])).long()
        self.nets.biased_classifier.train()

        return bias_score_array, debias_idx

    def confirm_pseudo_label_(self, bias_score_array, debias_idx, total_num):
        pseudo_label = (bias_score_array > self.args.tau).long()
        debias_label = torch.zeros(total_num).to(self.device)

        for idx in debias_idx:
            debias_label[idx] = 1

        spur_precision = torch.sum(
                (pseudo_label == 1) & (debias_label == 1)
            ) / torch.sum(pseudo_label)
        print("Spurious precision", spur_precision)
        spur_recall = torch.sum(
                (pseudo_label == 1) & (debias_label == 1)
            ) / torch.sum(debias_label)
        print("Spurious recall", spur_recall)

        wrong_idx_path = ospj(self.args.checkpoint_dir, 'wrong_index.pth')

        if not self.args.supervised:
            torch.save(pseudo_label, wrong_idx_path)
        else:
            torch.save(debias_label, wrong_idx_path)
        print('Saved pseudo label.')
        self.nets.classifier.train()
        self.nets.biased_classifier.train()

    def validation(self, fetcher, which='main'):
        if which == 'main':
            local_classifier = self.nets.classifier
        else:
            local_classifier = self.nets.biased_classifier
        local_classifier = local_classifier.eval()

        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)
        iterator = enumerate(fetcher)

        total_correct, total_num = 0, 0

        for index, (_, data, attr, fname) in iterator:
            label = attr[:, 0].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                logit = local_classifier(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

                total_correct += correct.sum()
                total_num += correct.shape[0]

            attr = attr[:, [0, 1]]
            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        print(attrwise_acc_meter.cum.view(self.attr_dims[0], -1))
        print(attrwise_acc_meter.cnt.view(self.attr_dims[0], -1))

        total_acc = total_correct / float(total_num)
        accs = attrwise_acc_meter.get_mean()

        local_classifier = local_classifier.train()
        return total_acc, accs

    def report_validation(self, valid_attrwise_acc, valid_acc,
                          step=0, which='bias', save_in_result=False):
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
        if save_in_result:
            with open(os.path.join(self.args.result_dir, 'test.txt'), "a") as f:
                f.write(log)

    def train_ERM(self, iters):
        logging.info('=== Start training ===')
        args = self.args
        nets = self.nets
        optims = self.optims

        fetcher = InputFetcher(self.loaders.train)
        fetcher_val = self.loaders.val
        fetcher_train = self.loaders.train

        total_num = len(self.loaders.trainset)
        bias_score_array = torch.zeros(total_num).to(self.device)
        pseudo_every = int(total_num / args.batch_size)

        start_time = time.time()

        self._save_checkpoint(step=0, token='initial')

        for i in range(iters):
            # fetch images and labels
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname

            pred = self.nets.classifier(x)
            pred_bias = self.nets.biased_classifier(x)

            loss = self.criterion(pred, label).mean()
            if args.pseudo_label_method == 'ensemble':
                loss_bias = self.criterion(pred_bias, label)
                bias_prob = nn.Softmax()(pred_bias)[torch.arange(pred_bias.size(0)), label]
                loss_bias = loss_bias[bias_prob > args.eta].mean() # Choose samples with high confidence
            else:
                if args.select_with_GCE:
                    loss_bias = self.bias_criterion(pred_bias, label).mean()
                else:
                    loss_bias = self.criterion(pred_bias, label).mean()

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

            if (i+1) % args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='main')
                self.valid_logger.append(total_acc.item(), which='ERM')

                total_acc_b, valid_attrwise_acc_b = self.validation(fetcher_val, which='bias')
                self.report_validation(valid_attrwise_acc_b, total_acc_b, i, which='bias')

            if (i+1) % pseudo_every == 0:
                bias_score_array, debias_idx = self.update_pseudo_label(bias_score_array, fetcher_train, iters, pseudo_every)

            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1, token='pretrain')

            if not self.args.no_lr_scheduling:
                self.scheduler.classifier.step()
                self.scheduler.biased_classifier.step()

        if args.pseudo_label_method == 'ensemble':
            self.confirm_pseudo_label_(bias_score_array, debias_idx, total_num)

        self.valid_logger.save()

        # save model checkpoints
        self._save_checkpoint(step=i+1, token='pretrain')

    def train(self):
        self.train_ERM(self.args.pretrain_iter)

    def _tsne(self, loader):
        # Plot t-SNE of hidden feature
        loader_iter = enumerate(loader)
        img = torch.empty(0).to(self.device)
        label = torch.empty(0).to(self.device)
        bias = torch.empty(0).to(self.device)

        for i, val in loader_iter:
            _, data, attr, _ = val
            label = torch.cat([label,
                               attr[:, 0].to(self.device)])
            bias = torch.cat([bias,
                              attr[:, 1].to(self.device)])
            img = torch.cat([img,
                             data.to(self.device)])
            if i > 1: break
        label = label.data.cpu().numpy()
        bias = bias.data.cpu().numpy()

        h = self.nets.classifier.extract(img)
        tsne = self.tsne.fit_transform(h.data.cpu().numpy())

        sample_path = lambda x: os.path.join(self.args.log_dir, f'{x}-tSNE.jpg')
        utils.plot_embedding(tsne, label, sample_path('class-aligned'))
        utils.plot_embedding(tsne, bias, sample_path('bias-aligned'))

        with open(ospj(self.args.log_dir, 'tSNE.pkl'), 'wb') as f:
            tsne_dict = {
                'tsne': tsne,
                'label': label,
                'bias': bias
            }
            pkl.dump(tsne_dict, f)

