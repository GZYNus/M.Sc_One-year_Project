import torch
import torch.nn as nn


# class LSTMcell(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layer):
#         super(LSTMcell, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, input):
#         x, _ = self.lstm(input)
#         s, b, h = x.shape
#         x = x.view(s*b, h)
#         output = self.fc(x)
#         return output


class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMcell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.output_size = output_size
        # self.phy_gate = nn.Linear(self.input_size,self.hidden_size)
        self.gate = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        # self.fc = nn.Linear(3* self.hidden_size, self.hidden_size)
        # self.output = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, input, h_t, c_t):
        # z = self.phy_gate(input)
        combined = torch.cat((input, h_t), 1)
        f_gate = self.gate(combined)
        i_gate = self.gate(combined)
        o_gate = self.gate(combined)

        f_gate = self.sigmoid(f_gate)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        g = self.gate(combined)
        # g = self.fc(g)
        g = self.tanh(g)
        c_t = torch.add(torch.mul(c_t, f_gate), torch.mul(g, i_gate))
        h_t = torch.mul(self.tanh(c_t), o_gate)
        # output = self.output(h_t)
        return h_t, c_t


