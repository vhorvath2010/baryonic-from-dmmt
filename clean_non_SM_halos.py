import torch
import numpy as np
from torch_geometric.data import Data

# Load pruned graphs with baryonic info
print("Loading graphs...")
graphs = torch.load("SG256_Full_Graphs.pt")

# Special parsing because np test/train/split
if not isinstance(graphs, list) or not isinstance(graphs[0], Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

# Remove halos with <= 0 stellar mass
cleaned_graphs = []
print(f"Cleaning {len(graphs)} graphs...")

# Remove halos with parents (subhalos)
subhalos = torch.load('SG256_subhalos.pt')

# Store subhalo by redshift
subhalo_map = {}
for subhalo in subhalos:
    rs = round(subhalo[1].item(), 4)
    subhalos_at_rs = subhalo_map.get(rs, [])
    subhalos_at_rs.append(subhalo)
    subhalo_map[rs] = subhalos_at_rs

def is_subhalo(halo_x):
    rs = round(halo_x[1].item(), 4)
    # if subhalo_map.get(rs) is None:
        # print("RS found with no subhalos:", rs)
    for subhalo in subhalo_map.get(rs, []):
        subhalo_pos = subhalo[0]
        if subhalo_pos[0:3] == halo_x[2:5].tolist():
            return True
    return False


for graph in graphs:
    # find indices without -1 for stellar mass
    valid_halo_idxs = np.where(graph.y > 0)[0]
    valid_halo_idxs = torch.from_numpy(valid_halo_idxs)
    # print("Halos with SM:", valid_halo_idxs)
    
    # find indices of those with no parents
    non_subhalo_mask = np.array([0 if is_subhalo(halo_x) else 1 for halo_x in graph.x]) 
    non_subhalo_idxs = np.where(non_subhalo_mask)
    # print("Non subhalos:", non_subhalo_idxs) 

    # ensure valid indices have SM and are not subhalos
    before_subhalo_prune = len(valid_halo_idxs)
    valid_halo_idxs = torch.from_numpy(np.intersect1d(valid_halo_idxs, non_subhalo_idxs))
    print('subhalo pruning removed', before_subhalo_prune - len(valid_halo_idxs), 'halos')

    # create subgraph with those halos
    cleaned_graph = graph.subgraph(valid_halo_idxs)

    # alert if any halos were cut
    # invalid = len(graph.y) - len(valid_halo_idxs) 
    # if invalid > 0:
    #    print(f"{invalid} invalid halos found for this graph!")

    # only append if there are any halos left
    if len(cleaned_graph.y) > 0:
        cleaned_graphs.append(cleaned_graph)

print("Saving cleaned graphs...")
torch.save(cleaned_graphs, "SG256_Full_ONLY_SM_NO_SUBHALO.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
