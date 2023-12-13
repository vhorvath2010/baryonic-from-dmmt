import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys

# Get training data path as argument
if len(sys.argv) != 2:
    print("Invalid Arguments!\nUsage: python train_gnn.py %training_data_path%")
    exit()

train_data_loc = sys.argv[1]

graphs = torch.load(train_data_loc)

# Special parsing because np test/train/split
if not isinstance(graphs, Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

print(f'loaded {len(graphs)} graphs')

# Select specific features for each data (currently mass and redshift)
for graph in graphs:
    graph.x = torch.tensor([[data[0], data[1]] for data in graph.x])
    
valid = True
for graph in graphs:
    if not graph.validate():
        valid = False
print(f'Graphs are valid?: {valid}')
print(f'input shape for graph 0: {graphs[0].x.shape}, output shape: {graphs[0].y.shape}')
if not valid:
    print("Invalid graphs found! Exiting...")
    exit(1)

# Model arch
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

# Custom loss function to ignore non-zero SM values
# from torchmetrics import MeanAbsolutePercentageError

class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        non_zero_mask = targets != 0
        non_zero_predictions = torch.where(non_zero_mask, predictions, torch.zeros_like(predictions))
        non_zero_targets = torch.where(non_zero_mask, targets, torch.zeros_like(targets))
        loss = (non_zero_predictions - non_zero_targets) ** 2
        return torch.mean(loss)

# Create DataLoader for batching
loader = DataLoader(graphs, batch_size=64, shuffle=True)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device {device}...")

model = GCN().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss().to(device)

best_state = None
best_loss = float('inf')

epochs = 500
print("Starting training...")
# Track average losses
avg_losses = []
for epoch in range(1, epochs + 1):
    total_loss = 0
    print(f"Epoch {epoch}...")
    for batch in loader:
        y_hat = torch.tensor([]).to(device)
        y = torch.tensor([]).to(device)

        # Run predictions on all graphs in batch before doing loss
        data = batch.to(device)
        out = model(data)
        y_hat = torch.cat((y_hat, out))
        y = torch.cat((y, data.y))
    
        # Filter out zero-mass truths from loss gradient calculations
        # non_zero_mask = y != 0
        # non_zero_predictions = torch.squeeze(y_hat)[non_zero_mask]
        # non_zero_truth = y[non_zero_mask]
        loss = loss_fn(torch.squeeze(y_hat), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    avg_losses.append(avg_loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = model.state_dict()
    if epoch == 1 or epoch % 5 == 0:
        print(f"Loss on epoch {epoch}: {avg_loss}")
        
# Save model
model_name = 'model_11_1_23'
torch.save(best_state, f'models/{model_name}.pt')
torch.save(avg_losses, f'models/{model_name}_losses.pt')
print(f"Best model saved with loss {best_loss}")
