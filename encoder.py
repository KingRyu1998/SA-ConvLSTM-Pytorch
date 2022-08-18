# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/16 10:38
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : decoder.py
@Language: Python3
'''

import torch
from torch import nn
from links import interlinkage

class Encoder(nn.Module):

     def __init__(self, encoder_params):
         super(Encoder, self).__init__()
         links = encoder_params[0]
         rnns = encoder_params[1]
         assert len(links) == len(rnns)
         self.blocks = len(rnns)
         for idx, (link, rnn) in enumerate(zip(links, rnns)):
             setattr(self, 'link' + str(idx), interlinkage(link))
             setattr(self, 'rnn' + str(idx), rnn)

     def forward_by_step(self, inputs, link, rnn):
         B, S, C, H, W = inputs.size()
         inputs = torch.reshape(inputs, (-1, C, H, W))
         inputs = link(inputs)
         inputs = torch.reshape(inputs, (B, S, inputs.size(1), inputs.size(2), inputs.size(3)))
         outputs, hidden_state = rnn(inputs)
         return outputs, hidden_state

     def forward(self, inputs):
         hidden_states = []
         for idx in range(self.blocks):
             inputs, hidden_state = self.forward_by_step(inputs,
                                           getattr('link' + str(idx)),
                                           getattr('rnn' + str(idx)))
             hidden_states.append(hidden_state)
         return tuple(hidden_states)
