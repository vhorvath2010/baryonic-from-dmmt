import torch
import numpy as np
from torch_geometric.data import Data

# Load graphs with baryonic info
print("Loading graphs...")
graphs = torch.load("datasets/unpruned/SG256_Full.pt")

# Ensure we're dealing with a list of graphs, otherwise assume it's in array form
if not isinstance(graphs, list) or not isinstance(graphs[0], Data):
    graphs = [Data(x=g[0][1], edge_index=g[1][1], y=g[2][1]) for g in graphs]

# OPTIONAL: load subhalos (to exclude)
# subhalos = torch.load("datasets/SG256_subhalos.pt")

# Store subhalo by redshift
# subhalo_map = {}
# for subhalo in subhalos:
#    rs = round(subhalo[1].item(), 4)
#    subhalos_at_rs = subhalo_map.get(rs, [])
#    subhalos_at_rs.append(subhalo)
#    subhalo_map[rs] = subhalos_at_rs


# def is_subhalo(halo_x):
#    rs = round(halo_x[1].item(), 4)
# if subhalo_map.get(rs) is None:
# print("RS found with no subhalos:", rs)
#    for subhalo in subhalo_map.get(rs, []):
#        subhalo_pos = subhalo[0]
#        if subhalo_pos[0:3] == halo_x[2:5].tolist():
#            return True
#    return False


# Remove halos based on stellar mass
cleaned_graphs = []
print(f"Cleaning {len(graphs)} graphs...")
for graph in graphs:
    # debugging, check if any halos have any stellar mass, if so include the graph
    # if max(graph.y) > 0:
    #     cleaned_graphs.append(graph)
    #     continue
    # find indices without valid SM

    # Stellar mass thresholding
    valid_mask = (graph.y > 0) & (graph.y < 10**4)

    # OPTIONAL: Prune subhalos
    # Find subhalos
    # non_subhalo_mask = np.array([0 if is_subhalo(halo_x) else 1 for halo_x in graph.x])
    # non_subhalo_idxs = np.where(non_subhalo_mask)
    # Prune them
    # before_subhalo_prune = len(valid_halo_idxs)
    # valid_halo_idxs = torch.from_numpy(
    #     np.intersect1d(valid_halo_idxs, non_subhalo_idxs)
    # )
    # print(
    #     "subhalo pruning removed", before_subhalo_prune - len(valid_halo_idxs), "halos"
    # )

    # OPTIONAL: Cut halos where SM > DM
    # DM_values = np.array([x[0] for x in graph.x])
    # SM_values = np.array(graph.y)
    # valid_mask = SM_values <= DM_values
    # valid_mass_idxs = np.where(valid_mask)[0]

    # valid_halo_idxs = torch.from_numpy(np.intersect1d(valid_halo_idxs, valid_mass_idxs))

    # Get valid halo indices from mask
    valid_halo_idxs = torch.flatten(valid_mask.nonzero())

    # Get subgraph induced by valid halo indices
    cleaned_graph = graph.subgraph(valid_halo_idxs)

    # alert if any halos were cut
    # invalid = len(graph.y) - len(valid_halo_idxs)
    # if invalid > 0:
    #    print(f"{invalid} invalid halos found for this graph!")

    # only append if there are any halos left
    if len(cleaned_graph.y) > 0:
        cleaned_graphs.append(cleaned_graph)

print("Saving cleaned graphs...")
torch.save(cleaned_graphs, "datasets/low_range/SG256_0_SM_4.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
