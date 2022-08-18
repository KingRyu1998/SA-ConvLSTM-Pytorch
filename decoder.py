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

class Decoder(nn.Module):

     def __init__(self, decoder_params, seq_len):  # seq_len为需要预测的时间长度
         super(Decoder, self).__init__()
         links = decoder_params[0]
         rnns = decoder_params[1]
         assert len(links) == len(rnns)
         self.blocks = len(rnns)
         self.seq_len = seq_len
         for idx, (link, rnn) in enumerate(zip(links, rnns)):
             setattr(self, 'link' + str(idx), interlinkage(link))
             setattr(self, 'rnn' + str(idx), rnn)

     def forward_by_step(self, inputs, link, rnn, hidden_state):
         B, S, C, H, W = inputs.size()
         inputs, hidden_state = rnn(inputs, hidden_state=hidden_state, seq_len=self.seq_len)
         inputs = torch.reshape(inputs, (-1, C, H, W))
         outputs = link(inputs)
         outputs = torch.reshape(outputs, (B, S, inputs.size(1), inputs.size(2), inputs.size(3)))

         return outputs, hidden_state

     def forward(self, hidden_states):
         inputs = self.forward_by_step(None,
                                       getattr(self, 'link0'),
                                       getattr(self, 'rnn0'),
                                       hidden_states[-1],
                                       )
         for idx in range(1, self.blocks):
             inputs, hidden_state = self.forward_by_step(inputs,
                                           getattr(self, 'link' + str(idx)),
                                           getattr(self, 'rnn' + str(idx)),
                                           hidden_state[-idx - 1],
                                                         )
         return inputs
