from training.solver import Solver
import torch
from munch import Munch

class ERMSolver(Solver):
    # Hyperparameter settings in util/__init__.py
    def __init__(self, args):
        super(ERMSolver, self).__init__(args)

        for net in self.nets.keys():
            self.optims[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=0
            )

        self.scheduler = Munch()
        if args.do_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims[net], step_size=args.lr_decay_step, gamma=args.lr_gamma)

    def train(self):
        self.train_ERM(self.args.total_iter)
