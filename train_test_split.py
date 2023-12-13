import torch
import numpy as np
import math

# Load graphs
print("Loading graphs...")
graphs = torch.load("SG256_Full_ONLY_SM.pt")

# Sample into train, test, val split
graphs = np.array(graphs)

# Shuffle graphs up
print("Shuffling trees...")
np.random.shuffle(graphs)

# 15% test 15% val, rest to train
n_vt = math.ceil(0.15 * len(graphs))
n_train = len(graphs) - 2 * n_vt

# sample val and test
val = graphs[0:n_vt]
print(f"Picking {len(val)} graphs for val")

test = graphs[n_vt:2*n_vt]
print(f"Picking {len(test)} graphs for test")

train = graphs[2*n_vt:]
print(f"Picking {len(train)} graphs for training")

print("Saving test, train, val graphs...")
torch.save(val, "SG256_Full_ONLY_SM_val.pt")
torch.save(test, "SG256_Full_ONLY_SM_test.pt")
torch.save(train, "SG256_Full_ONLY_SM_train.pt")
