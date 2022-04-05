from abc import *
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.fc = nn.Linear(last_dim, num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

    @abstractmethod
    def extract(self, inputs):
        pass

    def forward(self, inputs, is_feature=False, penultimate=False, simclr=False):
        # No classification. self.fc will be employed in train()
        _aux = {}

        if not is_feature:
            features = self.extract(inputs)
        else:
            features = inputs

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        return _aux

