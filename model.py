# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/17 8:54
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : model.py
@Language: Python3
'''

import torch
from torch import nn

class Net(nn.Module):

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        state = self.encoder(inputs)
        output = self.decoder(state)
        return output

