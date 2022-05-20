import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image


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

class CIFAR10Dataset(CMNISTDataset):
    def __init__(self, root, name='cmnist', split='train', transform=None, conflict_pct=5):
        super(CIFAR10Dataset, self).__init__(root, name, split, transform, conflict_pct)

class bFFHQDataset(CMNISTDataset):
    def __init__(self, root, name='cmnist', split='train', transform=None, conflict_pct=5):
        super(bFFHQDataset, self).__init__(root, name, split, transform, conflict_pct)

        if split=='test':
            self.data = glob(os.path.join(root, self.name, 'test', "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            self.data = data_conflict

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])
