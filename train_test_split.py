import torch
import numpy as np
import math

# Load graphs
print("Loading graphs...")
graphs = torch.load("datasets/unpruned/SG256_SM_Only.pt")

# Sample into train, test, val split
print(f"Loaded dataset with {len(graphs)} merger trees")

# Shuffle graphs up
print("Shuffling trees...")
np.random.shuffle(graphs)

# 15% test 15% val, rest to train
n_vt = math.floor(0.10 * len(graphs))
n_train = len(graphs) - 2 * n_vt

# sample val and test
val = graphs[0:n_vt]
print(f"Selected {len(val)} graphs for val")
print(f"Val graphs have {np.sum([len(graph.x) for graph in val])} total halos")

test = graphs[n_vt : 2 * n_vt]
print(f"Selected {len(test)} graphs for test")
print(f"Test graphs have {np.sum([len(graph.x) for graph in test])} total halos")

train = graphs[2 * n_vt :]
print(f"Selected {len(train)} graphs for training")
print(f"Training graphs have {np.sum([len(graph.x) for graph in train])} total halos")

print("Saving test, train, val graphs...")
torch.save(val, "datasets/unpruned/SG256_SM_Only_Val.pt")
torch.save(test, "datasets/unpruned/SG256_SM_Only_Test.pt")
torch.save(train, "datasets/unpruned/SG256_SM_Only_Train.pt")
