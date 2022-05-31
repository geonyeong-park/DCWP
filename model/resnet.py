import torchvision
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.model_zoo import load_url

from typing import Type, Any, Callable, Union, List, Optional
from prune.GateResnet import URL_DICT


class ResNet_(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet_, self).__init__(block, layers, num_classes, zero_init_residual,
                                      groups, width_per_group, replace_stride_with_dilation,
                                      norm_layer)
        self.last_dim = 512 * block.expansion

    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        return feature

def ResNet18(IMAGENET_pretrained=False):
    net = ResNet_(BasicBlock, [2, 2, 2, 2])
    if IMAGENET_pretrained:
        url = URL_DICT['resnet18']
        print(f'Load {url}')
        checkpoint = load_url(url)
        net.load_state_dict(checkpoint)
    return net

def ResNet34(IMAGENET_pretrained=False):
    net = ResNet_(BasicBlock, [3, 4, 6, 3])
    if IMAGENET_pretrained:
        url = URL_DICT['resnet34']
        print(f'Load {url}')
        checkpoint = load_url(url)
        net.load_state_dict(checkpoint)
    return net
