import os
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils import data
from munch import Munch
from data.transforms import transforms, use_preprocess
from data.dataset import CMNISTDataset, CIFAR10Dataset, bFFHQDataset, CUBDataset, \
    BARDataset, CelebADataset, IdxDataset


dataset_name_dict = {'cifar10c': CIFAR10Dataset,
                     'cmnist': CMNISTDataset,
                     'bffhq': bFFHQDataset,
                     'bar': BARDataset,
                     'cub': CUBDataset,
                     'celebA': CelebADataset}

def get_original_loader(args, return_dataset=False, sampling_weight=None):
    dataset_name = args.data
    transform = transforms['preprocess' if use_preprocess[dataset_name] else 'original'][dataset_name]['train']
    dataset_class = dataset_name_dict[dataset_name]

    dataset = dataset_class(root=args.train_root_dir, name=dataset_name, split='train',
                            transform=transform, conflict_pct=args.conflict_pct)
    if return_dataset:
        return dataset
    else:
        dataset = IdxDataset(dataset)
        if sampling_weight is not None:
            #TODO: replace?
            sampler = WeightedRandomSampler(sampling_weight, args.batch_size, replacement=True)
            return data.DataLoader(dataset=dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   sampler=sampler,
                                   pin_memory=True)
        else:
            return data.DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

def get_val_loader(args, split='test'):
    dataset_name = args.data
    transform = transforms['preprocess' if use_preprocess[dataset_name] else 'original'][dataset_name]['test']
    dataset_class = dataset_name_dict[dataset_name]

    dataset = dataset_class(root=args.val_root_dir, split=split, transform=transform)
    dataset = IdxDataset(dataset)
    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=True)

class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch(self):
        try:
            idx, x, attr, fname = next(self.iter) # attr: (class_label, bias_label)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            idx, x, attr, fname = next(self.iter)
        return idx, x, attr, fname

    def __next__(self):
        idx, x, attr, fname = self._fetch()
        y = attr[:, 0]
        bias_label = attr[:, 1]

        inputs = Munch(index=idx, x=x, y=y, bias_label=bias_label, fname=fname)

        return Munch({k: v if 'fname' in k else v.to(self.device)
                      for k, v in inputs.items()})

