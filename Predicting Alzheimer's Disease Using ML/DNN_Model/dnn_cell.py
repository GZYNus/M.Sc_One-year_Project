#!/usr/bin/env python
"""
Description: this is the basic cell of DNN, which includes one fully-connected layer, batch normalization and Xavier initialization method.
Email: gzynus@gmail.com
Author: Guo Zongyi
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.9, affine= True, track_running_stats= True)  # default: eps = 1e-05
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input):
        #hidden = F.relu(self.fc(input))
        out = self.fc(input)
        out = self.bn(out)
        hidden = F.relu(out)
        return hidden

