# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-callable
# pylint: disable=useless-import-alias
# pylint: disable=arguments-differ
# pylint: disable=invalid-name
"""
Name: MinimalRnnNet
Description: This is the network I constructed via MinimalRnnCell
Author: ljan@CBIG
"""
# import packages
import torch
import torch.nn as nn
from model.minimal_rnn_cell import MinimalRNNCell


class MinimalRNNNet(nn.Module):
    """
    RNN network
    """
    def __init__(self, args):
        super(MinimalRNNNet, self).__init__()
        self.args = args
        self.DEVICE = torch.device('cuda')
        # sequential model for multiple layers MinimalRnnNet
        self.cells = torch.nn.ModuleList()
        for i in range(self.args.layers):  # 加入每一层的参数
            if i == 0:
                cell = MinimalRNNCell(
                    batch_size=self.args.batch_size,
                    input_size=self.args.input_size,
                    hidden_size=self.args.hidden_size)

            else:
                cell = MinimalRNNCell(
                    batch_size=self.args.batch_size,
                    input_size=self.args.hidden_size,
                    hidden_size=self.args.hidden_size)
            self.cells.append(cell)
        # layers for making prediction
        self.continuous_pred = nn.Linear(self.args.hidden_size,
                                         self.args.input_size - 1)
        self.diagnosis_prob = nn.Linear(self.args.hidden_size, 3)

    def forward(self, original_inputs, hiddens, dropout_masks):
        """
        Forward pass
        :param original_in puts: input matrix. in shape of [batch_size, input_size]
        :param hiddens: hidden state list, length(hiddens) = numLayer
        :return:
                1. diagnosis_probability(normalized),
                In shape of [Batch_size, 3], corresponding to [NC, MCI, AD]
                2. diagnosis_prediction, 0 for NC,  1 for MCI, 2 for AD
                3. continuous_prediction, the prediction for continuous features,
                4. hidden states list
        """
        # input dropout
        if self.args.isTraining:
            inputs = original_inputs.masked_fill(dropout_masks[0] == 0, 0)   #????
        else:
            inputs = original_inputs
        # multiple layers
        for i in range(self.args.layers):
            # first layer
            if i == 0:
                hiddens[i] = self.cells[i](inputs, hiddens[i])
                # recurrent dropout
                if i != (self.args.layers - 1) and self.args.isTraining:
                    hiddens[i] = hiddens[i].masked_fill(dropout_masks[i+1] == 0, 0)
            # other layers
            else:
                last_hidden_i = hiddens[i]
                current_hidden_i = self.cells[i](hiddens[i - 1], last_hidden_i)
                hiddens[i] = current_hidden_i
                # recurrent dropout
                if i != (self.args.layers - 1) and self.args.isTraining:
                    hiddens[i] = hiddens[i].masked_fill(dropout_masks[i+1] == 0, 0)
        # making prediction
        # g_{t+1} = W_g * h_t + g_t
        continuous_prediction = torch.add(
            self.continuous_pred(hiddens[-1]), original_inputs[:, 1:])
        # s_{t+1} = softmax(W_s * h_t)
        diagnosis_probability = torch.softmax(
            self.diagnosis_prob(hiddens[-1]), dim=1)
        diagnosis_prediction = torch.max(diagnosis_probability, 1)[1]
        # return result
        return diagnosis_probability, diagnosis_prediction, continuous_prediction, hiddens

    def net_init(self):
        """
        Hidden states initialization
        :return: a list for initialized hidden states
        """
        init_hiddens = []
        dropout_masks = []
        indropoutmask = torch.zeros(self.args.batch_size, self.args.input_size).bernoulli_(1 - self.args.indrop)
        dropout_masks.append(indropoutmask.to(self.DEVICE))
        for i in range(self.args.layers):
            init_hidden = self.cells[i].cell_init()
            init_hiddens.append(init_hidden)
            redropoutmask = torch.zeros(self.args.batch_size, self.args.hidden_size).bernoulli_(1 - self.args.redrop)
            dropout_masks.append(redropoutmask.to(self.DEVICE))

        return init_hiddens, dropout_masks
