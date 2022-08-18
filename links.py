# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/16 10:50
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : links.py
@Language: Python3
'''

from collections import OrderedDict
from torch import nn

def interlinkage(link_params):
    layers = []
    for layer_name, params in link_params.items():
        if 'upsample' in layer_name:
            layer = nn.UpsamplingNearest2d(size=params[0], scale_factor=params[1])
            layers.append((layer_name, layer))
        elif 'pool' in layer_name:
            layer = nn.MaxPool2d(params[0], params[1], params[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            deconv2d = nn.ConvTranspose2d(
                in_channels=params[0],
                out_channels=params[1],
                kernel_size=params[2],
                stride=params[3],
                padding=params[4])
            layers.append(('deconv', deconv2d))
            if 'relu' in layer_name:
                layers.append(('relu', nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky', nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(
                in_channels=params[0],
                out_channels=params[1],
                kernel_size=params[2],
                stride=params[3],
                padding=params[4]
            )
            layers.append(('conv', conv2d))
            if 'relu' in layer_name:
                layers.append(('relu', nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky', nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
        return nn.Sequential(OrderedDict(layers))



