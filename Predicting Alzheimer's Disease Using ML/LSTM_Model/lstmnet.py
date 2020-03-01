import torch
import torch.nn as nn
from model.lstm_cell import LSTMcell

class LSTMnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMnet, self).__init__()
        self.cells = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            if i == 0:
                self.cells.append(LSTMcell(input_size, hidden_size))
            else:
                self.cells.append(LSTMcell(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input, H_N, C_N):
        for i in range(self.num_layers):
            if i == 0:
                H_N[i], C_N[i] = self.cells[i](input, H_N[i], C_N[i])
            else:
                H_N[i], C_N[i] = self.cells[i](H_N[i-1], H_N[i-1], C_N[i-1])

        output = self.fc(H_N[i])
        return output


