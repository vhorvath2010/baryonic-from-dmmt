import torch
import numpy as np
import math

# Load graphs
print("Loading graphs...")
graphs = torch.load("datasets\low_range\SG256_0_SM_5.pt")

# Sample into train, test, val split
print(f"Loaded dataset with {len(graphs)} merger trees")

# Shuffle graphs up
print("Shuffling trees...")
np.random.shuffle(graphs)

# 20% test, rest to train
n_test = math.floor(0.20 * len(graphs))
n_train = len(graphs) - n_test

# sample val and test
# val = graphs[0:n_vt]
# print(f"Selected {len(val)} graphs for val")
# print(f"Val graphs have {np.sum([len(graph.x) for graph in val])} total halos")

test = graphs[0:n_test]
print(f"Selected {len(test)} graphs for test")
print(f"Test graphs have {np.sum([len(graph.x) for graph in test])} total halos")

train = graphs[n_test:]
print(f"Selected {len(train)} graphs for training")
print(f"Training graphs have {np.sum([len(graph.x) for graph in train])} total halos")

print("Saving test, train, val graphs...")
# torch.save(val, "datasets\low_range\SG256_0_SM_5_Val.pt")
torch.save(test, "datasets\low_range\SG256_0_SM_5_Test.pt")
torch.save(train, "datasets\low_range\SG256_0_SM_5_Train.pt")
