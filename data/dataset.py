import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd


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


class CUBDataset(Dataset):
    """
    CUB dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(self, root, name='cub', split='train', transform=None, conflict_pct=5):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = os.path.join(root, self.name, 'data/waterbird_complete95_forest2water2')

        # Read in metadata
        print(f"Reading '{os.path.join(self.header_dir, 'metadata.csv')}'")
        metadata_df = pd.read_csv(
            os.path.join(self.header_dir, 'metadata.csv'))
        self.metadata_df = metadata_df[metadata_df["split"] == self.split_dict[split]]

        self.confounder_array = self.metadata_df["place"].values
        self.y_array = self.metadata_df["y"].values
        self.filename_array = self.metadata_df["img_filename"].values

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        img_filename = os.path.join(self.header_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, img_filename


class BARDataset(Dataset):
    label_name = {
        'climbing': 0,
        'diving': 1,
        'fishing': 2,
        'pole vaulting': 3,
        'racing': 4,
        'throwing': 5
    }
    def __init__(self, root, name='bar', split='train', transform=None, conflict_pct=5):
        super(BARDataset, self).__init__()
        self.name = name
        self.transform = transform
        self.root = root

        self.header_dir = os.path.join(root, self.name, split)
        self.data = glob(os.path.join(self.header_dir, "*"))

        train_target_attr = []
        for data in self.data:
            fname = os.path.relpath(data, self.header_dir)
            label = BARDataset.label_name[fname.split('_')[0]]
            train_target_attr.append(label)
        self.y_array = torch.LongTensor(train_target_attr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # BARDataset does not have ground truth bias label. Replace with dummy -1 instead.
        attr = torch.LongTensor([int(self.y_array[index]), -1])

        img_filename = self.data[index]
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


