import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)
        self.conv5 = GCNConv(128, 128)
        
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
        
        x = self.leaky(x)

        return x
