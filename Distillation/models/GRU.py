import torch
import torch.nn as nn
import numpy as np
class GRU(nn.Module):
    def __init__(self,INPUT_SIZE, output_size):
        super(GRU, self).__init__()
        # self.LN1=nn.LayerNorm(INPUT_SIZE)
        #self.WINDOW_SIZE=WINDOW_SIZE
        self.INPUT_SIZE=INPUT_SIZE
        self.gru = nn.GRU(input_size=self.INPUT_SIZE,
                             hidden_size=256,
                             num_layers=2,
                             batch_first=True,
                             # dropout=0.5
                             )

        self.out = nn.Sequential(nn.Linear(256, output_size), nn.Softmax())

    def forward(self, x):
        x = x[:, np.newaxis, :]
        r_out, self.hidden = self.gru(x, None)  # x(batch,time_step,input_size)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out