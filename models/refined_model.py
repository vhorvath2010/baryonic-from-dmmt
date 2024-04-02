import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


# Model arch
class GCN(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_hidden, hidden_channels):
        super().__init__()
        self.gconv_layers = nn.ModuleList()
        self.fc_layers = nn.Sequential()

        # Add first SAGEConv block
        self.gconv_layers.append(SAGEConv(input_channels, hidden_channels))
        self.gconv_layers.append(nn.LeakyReLU())
        self.gconv_layers.append(nn.BatchNorm1d(hidden_channels))

        # Add hidden SAGEConv blocks
        for i in range(num_hidden):
            self.gconv_layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.gconv_layers.append(nn.LeakyReLU())
            # Batch normalization for all but the last layer
            if i != num_hidden - 1:
                self.gconv_layers.append(nn.BatchNorm1d(hidden_channels))

        # Add FC layers
        self.fc_layers.append(nn.Linear(hidden_channels, output_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply conv layers
        for layer in self.gconv_layers:
            if isinstance(layer, SAGEConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        # Apply FC layers
        out = self.fc_layers(x)

        return out
