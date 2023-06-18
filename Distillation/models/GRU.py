import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class GRU(nn.Module):
    # def __init__(self,INPUT_SIZE, output_size):
    #     super(GRU, self).__init__()
    #     # self.LN1=nn.LayerNorm(INPUT_SIZE)
    #     #self.WINDOW_SIZE=WINDOW_SIZE
    #     self.INPUT_SIZE = INPUT_SIZE
    #     self.gru = nn.GRU(input_size=self.INPUT_SIZE,
    #                          hidden_size=32,
    #                          num_layers=2,
    #                          batch_first=True,
    #                          dropout=0.8
    #                          )
    #
    #     self.out = nn.Sequential(nn.Linear(32, output_size), nn.Softmax())
    #
    # def forward(self, x):
    #     x = x[:, np.newaxis, :]
    #     r_out, self.hidden = self.gru(x, None)  # x(batch,time_step,input_size)
    #     # choose r_out at the last time step
    #     out = self.out(r_out[:, -1, :])
    #     return out

    def __init__(self, INPUT_SIZE, output_size, n_layers=2, hidden_dim=5):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(INPUT_SIZE, hidden_dim, n_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        out, h_n = self.gru(x)
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x

