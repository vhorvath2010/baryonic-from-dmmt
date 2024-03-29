import torch
import sys
import os

# Get file location and number of jobs from cmdline input
if len(sys.argv) != 3:
    print("Invalid Arguments!\nUsage: python combine_job_array_outputs.py %path_prefix% %n_jobs$")
    exit()

path_prefix = sys.argv[1]
n_jobs = int(sys.argv[2])

# Merge the y outputs from each partial graph
xs = None # Shape: (graphs, halos) eg xs[i] = x values for halo i
ys = None # Shape: (graphs, halos) eg ys[i] = y values for halo i
print(f"Merging outputs {path_prefix}0.pt to {path_prefix}{n_jobs-1}.pt...")
for i in range(n_jobs):
    # Ensure file exists 
    curr_path = path_prefix + str(i) + ".pt"
    if not os.path.isfile(curr_path):
        print("File:", curr_path, "not found! Skipping outputs")
        continue

    # Load graphs for this snapshot
    graphs = torch.load(curr_path)

    # Handle first case, if ys is none, just set it equal to this graphs ys
    if ys == None:
        ys = [graph.y for graph in graphs]
        continue
    if xs == None:
        xs = [graph.x for graph in graphs]
        continue

    # Otherwise, if a graph.y contains a non -1 value, replace the value in ys, and xs
    # Note: ys_update is the ys captured for the current snapshot
    snapshot_ys = [graph.y for graph in graphs]
    snapshot_xs = [graph.x for graph in graphs]
    for graph_i in range(len(ys)): # loop through each graph
        for halo_i in range(len(ys[graph_i])): # loop through each halo
            if ys[graph_i][halo_i] == -1 and snapshot_ys[graph_i][halo_i] != -1:
                ys[graph_i][halo_i] = snapshot_ys[graph_i][halo_i]
                xs[graph_i][halo_i] = snapshot_xs[graph_i][halo_i]
    print(f"Snapshot {i} loaded")

# Take first set of graphs and update with merged ys, save that
graphs = torch.load(path_prefix + "0.pt")
for i in range(len(graphs)):
    graphs[i].y = ys[i]
    graphs[i].x = xs[i]
print(f"Saving combined outputs to {path_prefix}Merged.pt")
torch.save(graphs, path_prefix + "Merged.pt")
