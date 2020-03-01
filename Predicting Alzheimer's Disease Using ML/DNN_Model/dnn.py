#!/usr/bin/env python
"""
Description: DNN Model
Email: gzynus@gmail.com
Author: Guo Zongyi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dnn_cell import DNNCell


class FrogDNN(nn.Module):
    def __init__(self, nb_layer, input_size, batch_size, pred_size,
                 hidden_size, i_ratio, h_ratio, dev, isTraining):
        super(FrogDNN, self).__init__()
        self.nb_layer = nb_layer
        self.input_size = input_size
        self.batch_size = batch_size
        self.pred_size = pred_size
        self.hidden_size = hidden_size
        self.i_ratio = i_ratio
        self.h_ratio = h_ratio
        self.isTraining = isTraining
        self.dev = dev
        self.cells = nn.ModuleList()
        # we add multiple layers in FrogDNN
        self.cells.append(DNNCell(input_size, hidden_size[0]))
        for i in range(1, nb_layer):
            self.cells.append(DNNCell(hidden_size[i-1],
                                      hidden_size[i]))
        # layer for making prediction
        self.hid2pred = nn.Linear(hidden_size[nb_layer - 1],
                                  pred_size)

    def dropout_masks(self):
        i_mask = torch.ones(
            self.batch_size, self.input_size, device=self.dev)
        h_masks = [
            torch.ones(self.batch_size, cell.hidden_size, device=self.dev)
            for cell in self.cells
        ]

        if self.isTraining:
            i_mask.bernoulli_(1 - self.i_ratio)
            for mask in h_masks[:self.nb_layer]:  # no need to mask
                # last hidden
                mask.bernoulli_(1 - self.h_ratio)

        return i_mask, h_masks

    def forward(self, input):
        # get masks
        i_mask, h_masks = self.dropout_masks()
        h = input.masked_fill(i_mask == 0, 0)  # masked input
        # multiple layers
        for h_mask, cell in zip(h_masks, self.cells):
            h = cell(h)
            h *= h_mask

        # making prediction
        pred = self.hid2pred(h)
        return pred

















