import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, INPUT_SIZE, output_size, n_layers=2, hidden_dim=256):
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
