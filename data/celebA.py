import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import os
from PIL import Image

from data_aug.view_generator import ContrastiveLearningViewGenerator
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA


class BiasedCelebASplit:
    def __init__(self, root, split, transform, target_attr, **kwargs):
        self.transform = transform
        self.target_attr = target_attr

        self.celeba = CelebA(
            root=root,
            split="train" if split == "train_valid" else split,
            target_type="attr",
            transform=transform,
            download=True
        )
        self.bias_idx = 20 # Gender

        if target_attr == 'blonde':
            # Read in attributes
            self.attrs_df = pd.read_csv(os.path.join(root, "metadata.csv"))

            # Split out filenames and attribute names
            self.data_dir = os.path.join(root, "celeba", "img_align_celeba")
            self.filename_array = self.attrs_df["image_id"].values
            self.split_array = self.attrs_df["split"].values


            self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
            self.attr_names = self.attrs_df.columns.copy()

            # Then cast attributes to numpy array and set them to 0 and 1
            # (originally, they're -1 and 1)
            self.attrs_df = self.attrs_df.values
            self.attrs_df[self.attrs_df == -1] = 0

            # Get the y values
            target_idx = self.attr_idx('Blond_Hair')
            self.targets = self.attrs_df[:, target_idx]

            confounder_idx = self.attr_idx('Male')
            self.biases = self.attrs_df[:, confounder_idx]

            #self.target_idx = 9

            self.split_token = 0 if split == "train" else 2
            mask = self.split_array == self.split_token

            num_split = np.sum(mask)
            indices = np.where(mask)[0]

            self.filename_array = self.filename_array[indices]
            self.targets = torch.tensor(self.targets[indices]).long()
            self.biases = torch.tensor(self.biases[indices]).long()
            self.indices = indices

            """
            if split in ['train', 'train_valid']:
                save_path = Path(root) / 'pickles' / 'blonde'
                if save_path.is_dir():
                    print(f'use existing blonde indices from {save_path}')
                    self.indices = pickle.load(open(save_path / 'indices.pkl', 'rb'))
                else:
                    self.indices = self.build_blonde()
                    print(f'save blonde indices to {save_path}')
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f'indices.pkl', 'wb'))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))
            """

        elif target_attr == 'makeup':
            self.target_idx = 18
            self.attr = self.celeba.attr
            self.indices = torch.arange(len(self.celeba))
        else:
            raise AttributeError

        if target_attr != 'blonde':
            if split in ['train', 'train_valid']:
                save_path = Path(f'clusters/celeba_rand_indices_{target_attr}.pkl')
                if not save_path.exists():
                    rand_indices = torch.randperm(len(self.indices))
                    pickle.dump(rand_indices, open(save_path, 'wb'))
                else:
                    rand_indices = pickle.load(open(save_path, 'rb'))

                num_total = len(rand_indices)
                num_train = int(0.8 * num_total)

                if split == 'train':
                    indices = rand_indices[:num_train]
                elif split == 'train_valid':
                    indices = rand_indices[num_train:]

                self.indices = self.indices[indices]
                self.attr = self.attr[indices]

            self.targets = self.attr[:, self.target_idx]
            self.biases = self.attr[:, self.bias_idx]

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                                          targets=self.targets,
                                                                                                          biases=self.biases)

        print(f'Use BiasedCelebASplit \n target_attr: {target_attr} split: {split} \n {self.confusion_matrix_org}')


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def build_blonde(self):
        biases = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(biases == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((biases == 0) & (targets == 0))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def __getitem__(self, index):
        if self.target_attr != 'blonde':
            img, _ = self.celeba.__getitem__(self.indices[index])
        else:
            img_filename = os.path.join(self.data_dir,
                                        self.filename_array[index])
            img = Image.open(img_filename)
            img = self.transform(img)

        target, bias = self.targets[index], self.biases[index]
        return img, target, bias, index

    def __len__(self):
        return len(self.targets)


def get_celeba(root, target_attr='blonde', split='train', simclr_aug=True,
               img_size=224):
    logging.info(f'get_celeba - split:{split}, aug: {simclr_aug}')

    if split == 'train':
        if simclr_aug:
            transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        else:
            transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    if simclr_aug:
        transform = ContrastiveLearningViewGenerator(transform)

    dataset = BiasedCelebASplit(
        root=root,
        split=split,
        transform=transform,
        target_attr=target_attr,
    ) # bias: gender

    return dataset

def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by
