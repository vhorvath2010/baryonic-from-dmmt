import torch
import numpy as np

# Load pruned graphs with baryonic info
print("Loading graphs...")
graphs = torch.load("array_outputs/SG256_Full_Graphs_Part_Merged.pt")

# Remove halos with -1 stellar mass (data not loaded)
cleaned_graphs = []
print(f"Cleaning {len(graphs)} graphs...")
for graph in graphs:
    # find indices without -1 for stellar mass
    valid_halo_idxs = np.where(graph.y != -1)[0]
    valid_halo_idxs = torch.from_numpy(valid_halo_idxs)

    # create subgraph with those halos
    cleaned_graph = graph.subgraph(valid_halo_idxs)

    # alert if any halos were cut
    invalid = len(graph.y) - len(valid_halo_idxs) 
    if invalid > 0:
        print(f"{invalid} invalid halos found for this graph!")

    # only append if there are any halos left
    if len(cleaned_graph.y) > 0:
        cleaned_graphs.append(cleaned_graph)

print("Saving cleaned graphs...")
torch.save(cleaned_graphs, "SG256_Full_Merged_Cleaned_Graphs.pt")
print(f"{len(cleaned_graphs)} cleaned graphs saved!")
