# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/15 19:17
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : SA_ConvLSTM_cell.py
@Language: Python3
'''

from torch import nn
import torch

class SAM(nn.Module):

    def __init__(self, input_feature):
        super(self).__init__()
        self.feature_num = input_feature
        self.h_conv = nn.Conv2d(self.feature_num, 3 * self.feature_num, kernel_size=1)
        self.m_conv = nn.Conv2d(self.feature_num, 2 * self.feature_num, kernel_size=1)
        self.z_conv1 = nn.Conv2d(2 * self.feature_num, 2 * self.feature_num)
        self.z_conv2 = nn.Conv2d(3 * self.feature_num, 3 * self.feature_num)

    def forward(self, ih, im):

        # 特征聚合
        B, C, H, W = ih.size()
        h_qkv = self.h_conv(ih)
        m_kv = self.m_conv(im)
        Qh, Kh, Vh = torch.split(h_qkv, self.feature_num, dim=1)
        Km, Vm = torch.split(m_kv, self.feature_num, dim=1)
        Qh = Qh.veiw(B, C, H * W).transpose(1, 2)
        Kh = Kh.veiw(B, C, H * W)
        Vh = Vh.veiw(B, C, H * W)
        Ah = torch.softmax(torch.bmm(Qh, Kh), dim=-1)
        Am = torch.softmax(torch.bmm(Qh, Km), dim=-1)
        Zh = torch.matmul(Vh, Ah)
        Zm = torch.matmul(Vm, Am)
        Z = self.z_conv1(torch.cat([Zh, Zm], dim=1)).veiw(B, C, H, W)
        combination = torch.cat([Z, ih], dim=1)

        # 记忆更新
        gates = self.z_conv2(combination)
        o, g, i = torch.split(gates, self.feature_num, dim=1)
        i = torch.sigmoid(i)
        om = torch.tanh(g) * i + (1 - i) * im
        oh = torch.sigmoid(o) * om

        return oh, om

class SA_CLSTM_cell(nn.Module):

    def __init__(self, in_channels, kernel_size, feature_num):
        super(self).__init__()
        self.feature_num = feature_num
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels + self.feature_num, 4 * self.feature_num,
                              self.kernel_size, 1, self.padding, bias=True),
            nn.GroupNorm(4 * self.feature_num // 32, 4 * self.feature_num))
        self.sam = SAM(self.feature_num)

    def forward(self, inputs=None, seq_len=None, hidden_state=None):
        if inputs is None:
            seq_len = seq_len
            inputs = torch.zeros((hidden_state.size(0), seq_len, hidden_state.size(1), hidden_state(2), hidden_state(3)))
            ih, ic, im = hidden_state
        elif hidden_state is None:
            ih = torch.zeros((inputs.size(0), self.feature_num, inputs.size(-2), inputs.size(-1))).cuda()
            ic = torch.zeros((inputs.size(0), self.feature_num, inputs.size(-2), inputs.size(-1))).cuda()
            im = torch.zeros((inputs.size(0), self.feature_num, inputs.size(-2), inputs.size(-1))).cuda()
            seq_len = inputs.size(1)

        inner_output = []
        for index in range(seq_len):
            input = inputs[:, index, :]
            combination = torch.cat((input, ih), 1)
            gates = self.conv(combination)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.feature_num, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            oc = ic * forgetgate + ingate * cellgate
            oh = outgate * torch.tanh(oc)
            oh, om = self.sam(oh, im)
            inner_output.append(oh)
            ih = oh
            ic = oc
            im = om
        return torch.stack(inner_output, dim=1), (oh, oc, im)

