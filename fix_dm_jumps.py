import torch_geometric
import torch_geometric.utils
import torch
import sys

# Get data path as argument
if len(sys.argv) != 2:
    print("Invalid Arguments!\nUsage: python fix_dm_jumps.py %data path%")
    exit()

graphs = torch.load(sys.argv[1])
fixed_graphs = []


def get_adjacent_nodes(node_idx, edge_index):
    # Find the edges where the node is the source or target
    mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)

    # Get the adjacent nodes from the edge_index
    adjacent_nodes = edge_index[:, mask][1 - (edge_index[0] == node_idx)]
    return adjacent_nodes


for graph in graphs:
    # Compute the indegree for each node
    indegree = torch.zeros(graph.num_nodes, dtype=torch.long)
    for _, dst in graph.edge_index.t():
        indegree[dst] += 1

    # Find leaf nodes (nodes with indegree 0)
    leaf_nodes = torch.nonzero(indegree == 0, as_tuple=False).squeeze()

    print(len(leaf_nodes))