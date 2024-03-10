import torch
from torch_geometric.nn import SAGEConv


# Model arch
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = SAGEConv(2, 32)
        self.conv2 = SAGEConv(32, 128)
        self.conv3 = SAGEConv(128, 128)
        self.conv4 = SAGEConv(128, 128)
        self.conv5 = SAGEConv(128, 128)

        self.lin = torch.nn.Linear(128, 1)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = self.conv4(x, edge_index)
        x = self.relu(x)

        x = self.conv5(x, edge_index)
        x = self.relu(x)

        x = self.lin(x)

        return x
