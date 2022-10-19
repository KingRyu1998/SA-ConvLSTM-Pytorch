# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/17 9:09
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : train.py
@Language: Python3
'''

import os
import torch
from encoder import Encoder
from decoder import Decoder
from net_params import encoder_params, decoder_params
from model import Net
from torch.utils.data import DataLoader
from data_iter import Data_iter
from torch import optim
from early_stopping import Early_stopping
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
import numpy as np

# config
random_seed = 1997
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_save_path = r'./model_save'
train_dir_path = r'./'
val_dir_path = r'./'
batch_size = 8
total_epoch = 100
learning_rate = 1e-4

train_iter = Data_iter(train_dir_path)
val_iter = Data_iter(val_dir_path)
train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_iter, batch_size=batch_size, shuffle=False)

def train():

    # 搭建网络
    encoder = Encoder(encoder_params).cuda()
    decoder = Decoder(decoder_params, seq_len=6).cuda()  # seq_len为decoder输出的预测长度
    net = Net(encoder, decoder)

    # 实例化优化器、损失函数以及模型并行相关
    device = ['cuda' if torch.cuda.is_available() else 'cpu']
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    loss_func = nn.MSELoss()

    if os.path.exists(os.path.join(model_save_path, 'checkpoint.pth.tar')):
        print('loading existing model...')
        model_info = torch.load(os.path.join(model_save_path, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['model_state'])
        optimizer = optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optim_state'])
        cur_epoch = model_info['epoch'] + 1
    elif os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
        cur_epoch = 0
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 实例化训练策略
    early_stopping = Early_stopping(patience=10)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)
    # 记录1个epoch中所有train_loss
    train_losses = []
    # 记录1个epoch中所有val_loss
    val_losses = []
    # 记录所有epoch中每个epoch的平均train_loss
    avg_train_losses = []
    # 记录所有epoch中每个epoch的平均val_loss
    avg_val_losses = []
    for epoch in range(cur_epoch, total_epoch):
        
        #################
        # train the model
        #################
        net.train()
        tq_bar = tqdm(train_loader, leave=False, total=len(train_loader))
        for inputs, labels in tq_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            loss = loss_func(preds, labels)
            train_loss = loss.item() / batch_size
            train_losses.append(train_loss)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            tq_bar.set_postfix(
                {'train_loss': '{:.6f}'.format(train_loss),
                 'epoch': '{:02d}'.format(epoch)}
            )

        ####################
        # validate the model
        ####################
        with torch.no_grad():
            net.eval()
            tq_bar = tqdm(val_loader, leave=False, total=len(val_loader))
            for inputs, labels in tq_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = net(inputs)
                loss = loss_func(preds, labels)
                val_loss = loss.item() / batch_size
                val_losses.append(val_loss)
                tq_bar.set_postfix(
                    {'val_loss': '{:.6f}'.format(val_loss),
                     'epoch': '{:02d}'.format(epoch)}
                )

        # 释放显存
        torch.cuda.empty_cache()

        avg_train_loss = torch.mean(train_losses)
        avg_val_loss = torch.mean(val_losses)
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)

        print_msg = (f'epoch:{epoch}/{total_epoch}' +
                     f'train_loss:{avg_train_loss:.6f}'+
                     f'val_loss:{avg_val_loss:.6f}')
        print(print_msg)

        model_dict = {
            'model_state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
            'epoch': epoch
        }
        early_stopping(model_dict, model_save_path, avg_val_loss.item(), epoch)

        if early_stopping.early_stopping:
            print('Early stopping')
            break

        pla_lr_scheduler.step(avg_val_loss)

        # 重置记录单个epoch的loss容器
        train_losses = []
        val_losses = []

