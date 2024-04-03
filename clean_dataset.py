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
subhalos = torch.load("datasets/SG256_subhalos.pt")

# Store subhalo by redshift and position
subhalo_map = {}

for subhalo in subhalos:
    rs = round(subhalo[1].item(), 4)
    pos = (round(subhalo[0][0], 4), round(subhalo[0][1], 4), round(subhalo[0][2], 4))
    if rs not in subhalo_map:
        subhalo_map[rs] = {}
    if pos not in subhalo_map[rs]:
        # if subhalo_map[rs].get(pos, False): Confirmed 4 sig figs is good enough for not halo overlap in pos/rs
        #     print("Halo overlap!")
        subhalo_map[rs][pos] = True  # Store True to indicate existence


def is_subhalo(halo_x):
    rs = round(halo_x[1].item(), 4)
    pos = (
        round(halo_x[2].item(), 4),
        round(halo_x[3].item(), 4),
        round(halo_x[4].item(), 4),
    )

    return subhalo_map.get(rs, {}).get(pos, False)  # Check if subhalo exists @ pos @ rs


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
    valid_mask = (graph.y > 0) & (graph.y < 10**5)

    # Leave if no valid stars
    before_subhalo_prune = torch.sum(valid_mask)
    if before_subhalo_prune == 0:
        continue

    # OPTIONAL: Prune subhalos
    # Find subhalos
    non_subhalo_mask = np.array([0 if is_subhalo(halo_x) else 1 for halo_x in graph.x])
    # Prune them
    valid_mask = valid_mask & non_subhalo_mask
    print(f"subhalo pruning removed: {before_subhalo_prune - torch.sum(valid_mask)}")

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
torch.save(cleaned_graphs, "datasets/low_range/SG256_0_SM_5.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
