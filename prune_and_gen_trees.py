import ytree
import torch
import torch_geometric

print('loading data...')
arbor = ytree.load('/storage/home/hhive1/jw254/data/SG256-v3/rockstar_halos/trees/tree_0_0_0.dat') # Set your arbor here
trees = list(arbor[:])

# Important info for loading baryonic properties
print(arbor.field_info['virial_radius'])
print(arbor.field_info['position'])

# Load trees
graphs = []
print('loading and pruning all trees')
for tree in trees:
    ids_in_graph = {}  # hold uid -> idx pairs
    x = []
    edge_index = []  # add edges as [parent, child] and transpose afterward

    # start from leaf nodes and keep track of last relevant node (node where there is a large change since the previous
    # one). skip all nodes in between relevant nodes
    for leaf in tree.get_leaf_nodes():
        curr = leaf
        last_relevant = curr  # track last seen node that saw measurable change
        while curr.descendent is not None:  # run to root (which has no descendent)
            curr = curr.descendent
            # relevant = descendent is root (need to capture final merger) OR 10% or more change in mass
            if curr.descendent is None or (abs(curr['mass'].value.item() - last_relevant['mass'].value.item()) / last_relevant['mass'].value.item()) > 0.1:
                # add halos to graph if not there
                if curr.uid not in ids_in_graph:
                    ids_in_graph[curr.uid] = len(ids_in_graph)
                    pos = curr['position'].value.tolist()
                    rVir = curr['virial_radius'].value
                    node_data = [curr['mass'].value.item(), curr['redshift'], pos[0], pos[1], pos[2], rVir.item()]
                    x.append(node_data)

                if last_relevant.uid not in ids_in_graph:
                    ids_in_graph[last_relevant.uid] = len(ids_in_graph)
                    pos = last_relevant['position'].value.tolist()
                    rVir = last_relevant['virial_radius'].value
                    node_data = [last_relevant['mass'].value.item(), last_relevant['redshift'], pos[0], pos[1], pos[2], rVir.item()]
                    x.append(node_data)

                edge = [ids_in_graph[last_relevant.uid], ids_in_graph[
                    curr.uid]]  # add edge from last relevant to curr (skipping non-relevant nodes in-between)
                if edge not in edge_index:
                    edge_index.append(edge)
                last_relevant = curr

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index).T
    graph = torch_geometric.data.Data(x=x, edge_index=edge_index)
    if not graph.validate():
        print("Invalid graph found!")
    else:
        graphs.append(graph)
        print(f"Graph #len(graphs) was saved!")

print(f"Saving {len(graphs)} graphs...")
torch.save(graphs, 'SG256_pruned.pt') # Set save location here
print("Graphs saved!")
print("X is form: [mass (MSun), redshift, x, y, z, rVir]")
