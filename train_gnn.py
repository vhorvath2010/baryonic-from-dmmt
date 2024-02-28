import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sys
from datetime import datetime
from model import GCN

# Get training data path as argument
if len(sys.argv) != 2:
    print("Invalid Arguments!\nUsage: python train_gnn.py %training_data_path%")
    exit()

train_data_loc = sys.argv[1]

graphs = torch.load(train_data_loc)

# Special parsing because np test/train/split
if not isinstance(graphs, list) or not isinstance(graphs[0], Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

print(f"loaded {len(graphs)} graphs")

# Select specific features for each data (currently mass and redshift)
for graph in graphs:
    graph.x = torch.tensor([[data[0], data[1]] for data in graph.x])

valid = True
for graph in graphs:
    if not graph.validate():
        valid = False
print(f"Graphs are valid?: {valid}")
print(
    f"input shape for graph 0: {graphs[0].x.shape}, output shape: {graphs[0].y.shape}"
)
if not valid:
    print("Invalid graphs found! Exiting...")
    exit(1)

# Custom loss function to ignore non-zero SM values
# from torchmetrics import MeanAbsolutePercentageError


class CustomMSELoss(torch.nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        non_zero_mask = targets != 0
        non_zero_predictions = torch.where(
            non_zero_mask, predictions, torch.zeros_like(predictions)
        )
        non_zero_targets = torch.where(
            non_zero_mask, targets, torch.zeros_like(targets)
        )
        loss = (non_zero_predictions - non_zero_targets) ** 2
        return torch.mean(loss)


# Create DataLoader for batching
loader = DataLoader(graphs, batch_size=64, shuffle=False)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}...")

model = GCN().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss().to(device)

best_state = None
best_loss = float("inf")

epochs = 250
print("Starting training...")

# Track average losses (MSE average of halos)
avg_losses = []
n_halos = sum([len(graph) for graph in graphs])

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
        loss = loss_fn(
            torch.squeeze(y_hat), y
        )  # Use a "diagonal" covariance matrix (since we have 1 feature this is just a 1x1 matrix with 1 value 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / n_halos
    avg_losses.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = model.state_dict()

    if epoch == 1 or epoch % 5 == 0:
        print(f"Loss on epoch {epoch}: {avg_loss}")

# Save model
model_name = 'unpruned_model' # "model_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
torch.save(best_state, f"models/{model_name}.pt")
torch.save(avg_losses, f"models/{model_name}_losses.pt")
print(f"Best model saved with loss {best_loss}")
