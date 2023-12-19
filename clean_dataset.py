import torch
import numpy as np
from torch_geometric.data import Data

# Load pruned graphs with baryonic info
print("Loading graphs...")
graphs = torch.load("array_outputs/SG256_Full_Graphs_Part_Merged.pt")

# Ensure we're dealing with a list of graphs, otherwise assume it's in array form
if not isinstance(graphs, list) or not isinstance(graphs[0], Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

# OPTIONAL: load subhalos (to exclude)
subhalos = torch.load("SG256_subhalos.pt")

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


# Remove halos with -1 stellar mass (data not loaded)
cleaned_graphs = []
print(f"Cleaning {len(graphs)} graphs...")
for graph in graphs:
    # find indices without valid SM
    # OPTIONAL: Change 0 to be a threshold value is want SM only
    valid_halo_idxs = np.where(graph.y >= 0)[0]
    valid_halo_idxs = torch.from_numpy(valid_halo_idxs)

    # OPTIONAL: Prune subhalos
    # Find subhalos
    non_subhalo_mask = np.array([0 if is_subhalo(halo_x) else 1 for halo_x in graph.x])
    non_subhalo_idxs = np.where(non_subhalo_mask)
    # Prune them
    before_subhalo_prune = len(valid_halo_idxs)
    valid_halo_idxs = torch.from_numpy(
        np.intersect1d(valid_halo_idxs, non_subhalo_idxs)
    )
    print(
        "subhalo pruning removed", before_subhalo_prune - len(valid_halo_idxs), "halos"
    )

    # OPTIONAL: Cut halos where SM > DM
    DM_values = [x[0] for x in graph.x]
    SM_values = graph.y
    valid_mass_idxs = []
    curr_idx = 0
    for dm, sm in zip(DM_values, SM_values):
        if sm < dm:
            valid_mass_idxs.append(curr_idx)
        valid_halo_idxs = torch.from_numpy(
            np.intersect1d(valid_halo_idxs, valid_mass_idxs)
        )
        curr_idx += 1

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
torch.save(cleaned_graphs, "SG256_Full_Merged_Cleaned_Graphs.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
