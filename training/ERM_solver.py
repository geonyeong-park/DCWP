from training.solver import Solver
import torch
from munch import Munch

class ERMSolver(Solver):
    # Hyperparameter settings in util/__init__.py
    def __init__(self, args):
        super(ERMSolver, self).__init__(args)

    def train(self):
        self.train_ERM(self.args.total_iter)

    def evaluate(self):
        fetcher_val = self.loaders.val
        self._load_checkpoint(self.args.pretrain_iter, 'pretrain')
        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(False)

        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0, which='Test', save_in_result=True)

        self._tsne(fetcher_val)
