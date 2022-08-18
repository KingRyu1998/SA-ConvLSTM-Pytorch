# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/18 10:30
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : data_iter.py
@Language: Python3
'''

import os

import torch
from torch import nn
import numpy as np

class Data_iter(nn.Module):

    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path

    def read_data(self, file_path):
        data = np.load(file_path)
        input = torch.from_numpy(data[:6])
        label = torch.from_numpy(data[6:])
        return input, label

    def __getitem__(self, idx):
        file_name_list = os.listdir(self.dir_path)
        file_path = os.path.join(self.dir_path, file_name_list[idx])
        input, label = self.read_data(file_path)
        return input, label

    def __len__(self):
        return len(os.listdir(self.dir_path))

        