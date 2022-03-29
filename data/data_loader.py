import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils import data
from torchvision import transforms as T
from munch import Munch
from data.transforms import transforms, use_preprocess
from data.dataset import CMNISTDataset, CIFAR10Dataset, bFFHQDataset
from glob import glob
from PIL import Image

dataset_name_dict = {'cifar10c': CIFAR10Dataset,
                     'cmnist': CMNISTDataset,
                     'bffhq': bFFHQDataset}

def get_original_loader(args, mode='unsup', return_dataset=False):
    dataset_name = args.data
    transform = transforms['preprocess' if use_preprocess[dataset_name] else 'original'][dataset_name]['train']
    dataset_class = dataset_name_dict[dataset_name]

    if mode == 'unsup':
        if args.use_unsup_data:
            dataset = dataset_class(root=args.train_root_dir, name=dataset_name, split='train', mode='unsup', use_unsup_data=True,
                                    transform=transform, labeled_ratio=args.labeled_ratio, conflict_pct=args.conflict_pct)
        else:
            return None # If you want to use unsupervised dataset, you should assign args.labeled_ratio < 1.

    elif mode == 'sup':
        if args.use_unsup_data:
            dataset = dataset_class(root=args.train_root_dir, name=dataset_name, split='train', mode='sup', use_unsup_data=True,
                                    transform=transform, labeled_ratio=args.labeled_ratio, conflict_pct=args.conflict_pct)
        else:
            dataset = dataset_class(root=args.train_root_dir, name=dataset_name, split='train', mode='sup', use_unsup_data=False,
                                    transform=transform, labeled_ratio=1., conflict_pct=args.conflict_pct)

    if return_dataset:
        return dataset
    else:
        return data.DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)

def get_concat_loader(args):
    # For VAE only. VAE can use both labeled and unlabeled data.
    unsup_dataset = get_original_loader(args, mode='unsup', return_dataset=True)
    sup_dataset = get_original_loader(args, mode='sup', return_dataset=True)

    if unsup_dataset is not None:
        concat_dataset = data.ConcatDataset([unsup_dataset, sup_dataset])
    else:
        concat_dataset = sup_dataset

    return data.DataLoader(dataset=concat_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True)

def get_aug_loader(args, mode='unsup'):
    # align (original, augmented) dataset
    pass

def get_val_loader(args):
    dataset_name = args.data
    transform = transforms['preprocess' if use_preprocess[dataset_name] else 'original'][dataset_name]['test']
    dataset_class = dataset_name_dict[dataset_name]

    dataset = dataset_class(root=args.val_root_dir, split='test', transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=True)

class InputFetcher:
    def __init__(self, loader_sup, loader_unsup=None, use_unsup_data=True, mode='augment'):
        self.loader_sup = loader_sup
        self.loader_unsup = loader_unsup
        self.use_unsup_data = use_unsup_data
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_sup(self):
        try:
            x, attr, fname = next(self.iter) # attr: (class_label, bias_label)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader_sup)
            x, attr, fname = next(self.iter)
        return x, attr, fname

    def _fetch_unsup(self):
        try:
            x, attr, fname = next(self.iter_unsup) # attr is not allowed to be used in training
        except (AttributeError, StopIteration):
            self.iter_unsup = iter(self.loader_unsup)
            x, attr, fname  = next(self.iter_unsup)
        return x, attr, fname

    def __next__(self):
        if self.mode == 'augment':
            x, attr, fname = self._fetch_sup()

            if self.use_unsup_data:
                x_u, attr_u, fname_u = self._fetch_unsup()
                inputs = Munch(x_sup=x, attr_sup=attr, fname_sup=fname,
                               x_unsup=x_u, attr_unsup=attr_u, fname_unsup=fname_u)
            else:
                inputs = Munch(x_sup=x, attr_sup=attr, fname_sup=fname)

        elif self.mode == 'debias':
            pass #TODO: align (original, augment) dataset and sampling

        elif self.mode == 'test':
            x, attr, fname = self._fetch_sup()
            inputs = Munch(x_sup=x, attr_sup=attr, fname_sup=fname)
        else:
            raise NotImplementedError

        return Munch({k: v if 'fname' in k else v.to(self.device)
                      for k, v in inputs.items()})

