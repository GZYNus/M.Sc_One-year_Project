# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-callable
# pylint: disable=useless-import-alias
# pylint: disable=arguments-differ
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
Name: MinimalRnnCell
Description: This is the basic MinimalRNN cell for network construction
Author: ljan@CBIG
"""
# import packages
import torch
import torch.nn as nn


# model construction
class MinimalRNNCell(nn.Module):
    """
    Inputs:
        1. batch_size: an integer, Input should be in shape of [Batch_size, Input_size]
        2. input_size: an integer
        3. hidden_size: an integer, Hidden state should be in shape of [Batch_size, Hidden_size]
    Outputs:
        1. hidden_o, Updated hidden state
    """

    def __init__(self, batch_size, input_size, hidden_size):
        """
        Input:
            hyper parameters
        """
        super(MinimalRNNCell, self).__init__()
        # assign hyper parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        # torch.cuda.set_(gpu_id)
        self.device = torch.device('cuda')
        # input transform layer, \Phi(W_x * x_t)
        self.phi_layer = nn.Linear(self.input_size, self.hidden_size)
        # U_h * h_{t} + U_z * z_{t}
        self.u_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # initializer
        nn.init.constant_(self.u_layer.bias, val=1)

    def forward(self, inputs, hidden_i):
        """
        Forward pass
        """
        # Computation within MinimalRnnCell
        z = torch.tanh(self.phi_layer(inputs))
        combined = torch.cat((z, hidden_i), 1)
        u = torch.sigmoid(self.u_layer(combined))
        hidden_o = u * hidden_i + (1 - u) * z
        # return updated hidden_state
        return hidden_o

    def cell_init(self):
        """
        Hidden state initialization
        :return: initialized hidden state
        """

        # hidden state initialization
        init_hidden = torch.zeros(self.batch_size,
                                  self.hidden_size).to(self.device)

        return init_hidden
