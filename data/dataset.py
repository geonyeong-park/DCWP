import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np


class CMNISTDataset(Dataset):
    def __init__(self, root, name='cmnist', split='train', transform=None, conflict_pct=5):
        super(CMNISTDataset, self).__init__()
        self.name = name
        self.transform = transform
        self.root = root
        if conflict_pct >= 1:
            conflict_pct = int(conflict_pct)
        self.conflict_token = f'{conflict_pct}pct'

        if split=='train':
            self.header_dir = os.path.join(root, self.name, self.conflict_token)
            self.align = glob(os.path.join(self.header_dir, 'align', "*", "*"))
            self.conflict = glob(os.path.join(self.header_dir, 'conflict',"*", "*"))

            self.data = self.align + self.conflict

            train_target_attr = []
            for data in self.data:
                fname = os.path.relpath(data, self.header_dir)
                train_target_attr.append(int(fname.split('_')[-2]))
            self.y_array = torch.LongTensor(train_target_attr)

        elif split == 'valid':
            self.header_dir = os.path.join(root, self.name, self.conflict_token)
            self.data = glob(os.path.join(self.header_dir, 'valid',"*","*"))

        elif split=='test':
            self.data = glob(os.path.join(root, self.name, 'test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index] # attr=(class_label, bias_label)


class CelebADataset(Dataset):
    """
    CelebA dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(self, root, name='celebA', split='train', transform=None, conflict_pct=5):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = os.path.join(root, self.name)
        self.data_dir = os.path.join(self.header_dir, "celeba", "img_align_celeba")

        print(f"Reading '{os.path.join(self.header_dir, 'metadata_blonde_subsampled.csv')}'")
        self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "metadata_blonde_subsampled.csv"))
        self.filename_array = self.attrs_df["image_id"].values
        self.split_array = self.attrs_df["split"].values

        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0


        target_idx = self.attr_idx('Blond_Hair')
        self.y_array = self.attrs_df[:, target_idx]

        confounder_idx = self.attr_idx('Male')
        self.confounder_array = self.attrs_df[:, confounder_idx]

        self.split_token = 0 if split == "train" else 2
        mask = self.split_array == self.split_token

        num_split = np.sum(mask)
        indices = np.where(mask)[0]

        self.filename_array = self.filename_array[indices]
        self.y_array = torch.tensor(self.y_array[indices]).long()
        self.confounder_array = torch.tensor(self.confounder_array[indices]).long()
        self.indices = indices


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, img_filename

class CIFAR10Dataset(CMNISTDataset):
    def __init__(self, root, name='cifar10c', split='train', transform=None, conflict_pct=5):
        super(CIFAR10Dataset, self).__init__(root, name, split, transform, conflict_pct)

class bFFHQDataset(CMNISTDataset):
    def __init__(self, root, name='bffhq', split='train', transform=None, conflict_pct=5):
        super(bFFHQDataset, self).__init__(root, name, split, transform, conflict_pct)
        if split=='test':
            self.data = glob(os.path.join(root, self.name, 'test', "*"))

        elif split=='valid':
            self.data = glob(os.path.join(root, self.name, 'valid', "*"))

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])

