import ytree
import torch
import torch_geometric

print("loading data...")
arbor = ytree.load(
    "/storage/home/hhive1/jw254/data/SG256-v3/rockstar_halos/trees/tree_0_0_0.dat"
)  # Set your arbor here
trees = list(arbor[:])

# Load trees
graphs = []
print("loading and pruning all trees")
for tree in trees:
    ids_in_graph = {}  # hold uid -> idx pairs
    uids_in_x = set()
    x = []
    edge_index = []  # add edges as [parent, child] and transpose afterward

    # start from leaf nodes (no ancestors)
    for leaf in tree.get_leaf_nodes():
        curr = leaf
        while curr.descendent is not None:  # run to root (which has no descendent)
            curr = curr.descendent
            # Check if mapping exist
            if curr.uid not in ids_in_graph:
                # Save id mapping
                ids_in_graph[curr.uid] = len(ids_in_graph)

            # Ensure node isn't already in graph (should never not be the case...)
            if curr.uid not in uids_in_x:
                # Save x values
                pos = curr["position"].value.tolist()
                rVir = curr["virial_radius"].value
                node_data = [
                    curr["mass"].value.item(),
                    curr["redshift"],
                    pos[0],
                    pos[1],
                    pos[2],
                    rVir.item(),
                ]
                x.append(node_data)
                uids_in_x.add(curr.uid)

            # Generate edge from ancestor
            for ancestor in curr.ancestors:
                # Check if mapping exist
                if ancestor.uid not in ids_in_graph:
                    # Save id mapping
                    ids_in_graph[ancestor.uid] = len(ids_in_graph)
                edge = [
                    ids_in_graph[ancestor.uid],
                    ids_in_graph[curr.uid],
                ]  # add edge from last relevant to curr (skipping non-relevant nodes in-between)
                if edge not in edge_index:
                    edge_index.append(edge)

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index).T
    graph = torch_geometric.data.Data(x=x, edge_index=edge_index)
    if not graph.validate():
        print("Invalid graph found!")
    else:
        graphs.append(graph)
        print(f"Graph #{len(graphs)} was saved!")

print(f"Saving {len(graphs)} graphs...")
torch.save(graphs, "datasets/unpruned/SG256.pt")  # Set save location here
print("Graphs saved!")
print("X is form: [mass (MSun), redshift, x, y, z, rVir]")
