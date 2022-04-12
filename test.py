import torch
import torch.nn as nn
import argparse


class EMA(nn.Module):
    def __init__(self, label_dict, num_classes=None, alpha=0.9):
        super(EMA, self).__init__()

        self.label_dict = label_dict
        self.alpha = alpha
        parameter = {}
        updated = {}
        for fname, v in label_dict.items():
            parameter[fname] = torch.tensor(0.).to(v.device)
            updated[fname] = torch.tensor(0.).to(v.device)
        self.parameter = parameter
        self.updated = updated
        self.num_classes = num_classes

    def update(self, data, fname, curve=None, iter_range=None, step=None):
        if curve is None:
            self.parameter[fname] = self.alpha * self.parameter[fname] + (1 - self.alpha * self.updated[fname]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[fname] = alpha * self.parameter[fname] + (1 - alpha * self.updated[fname]) * data
        self.updated[fname] = 1

    def max_loss(self, label):
        label_index = [fname for fname, v in self.label_dict.items() if v == label]
        params = []
        for l in label_index:
            params.append(self.parameter[l])

        return params.max()

e = EMA({'holy':torch.tensor(1), 'god':torch.tensor(2)})
print(e.state_dict())
e.state_dict().update(e.parameter)
print(e.state_dict())
print(e.parameter)
torch.save(e.parameter, 'test.pth')
t = torch.load('test.pth')
print(t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--data', type=str, default='cmnist',
                        choices=['cmnist', 'cifar10c', 'bffhq'])
    args = parser.parse_args()
    setattr(args, 'holy', 1)
    print(args.holy==1)
