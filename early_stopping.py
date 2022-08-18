# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/17 9:14
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : early_stopping.py
@Language: Python3
'''

import os
import torch
import numpy as np

class Early_stopping():

    def __init__(self, patience=7, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.early_stopping = False
        self.count = 0
        self.pre_loss = np.Inf

    def __call__(self, model_info, save_path, valid_loss, epoch):
        if self.pre_loss <= valid_loss:
            self.count += 1
            if self.verbose:
                print(f'early_stopping counter: {self.count} out of {self.patience}')
        elif self.pre_loss > valid_loss:
            self.save_model(model_info, save_path, valid_loss, epoch)
        if self.count >= self.patience:
            self.early_stopping = True

    def save_model(self, model_info, save_path, valid_loss, epoch):
        if self.verbose:
            print(f'valid_loss decreased: {self.pre_loss:.6f} --> {valid_loss:.6f}. Saving model...')
        torch.save(
            model_info,
            os.path.join(save_path, 'checkpoint_{}_{:.6f}.pth.tar'.format(epoch, valid_loss)))
        self.pre_loss = valid_loss

