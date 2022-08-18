# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/16 10:53
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SA-ConvLSTM
@File    : net_params.py
@Language: Python3
'''

from collections import OrderedDict
from SA_ConvLSTM_cell import SA_CLSTM_cell

encoder_params = [
    [
        OrderedDict(
            {'pool': [4, 4, 0],
            'conv_leaky': [1, 16, 3, 1, 1]}
        ),
        OrderedDict({'conv_leaky': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv_leaky': [96, 96, 3, 2, 1]})
    ],

    [
        SA_CLSTM_cell(in_channels=16, feature_num=64, kernel_size=5),
        SA_CLSTM_cell(in_channels=64, feature_num=96, kernel_size=5),
        SA_CLSTM_cell(in_channels=96, feature_num=96, kernel_size=5)
    ]
]

decoder_params = [
    [
        OrderedDict({'deconv_leaky': [96, 96, 3, 2, 1]}),
        OrderedDict({'deconv_leaky': [96, 96, 3, 2, 1]}),
        OrderedDict(
                {'conv_leaky': [64, 16, 3, 1, 1],
                 'conv_leaky': [16, 1, 1, 1, 0],
                 'upsample': [(200, 200), 4]}
        )
    ],

    [
        SA_CLSTM_cell(in_channels=96, feature_num=96, kernel_size=5),
        SA_CLSTM_cell(in_channels=96, feature_num=96, kernel_size=5),
        SA_CLSTM_cell(in_channels=96, feature_num=64, kernel_size=5)
    ]
]