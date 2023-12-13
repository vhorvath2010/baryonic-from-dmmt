import torch
import sys
import os

# Get file location and number of jobs from cmdline input
if len(sys.argv) != 3:
    print("Invalid Arguments!\nUsage: python combine_job_arrray_outputs.py %path_prefix% %n_jobs$")
    exit()

path_prefix = sys.argv[1]
n_jobs = int(sys.argv[2])

# Merge the y outputs from each partial graph
ys = None
print(f"Merging outputs {path_prefix}0.pt to {path_prefix}{n_jobs-1}.pt...")
for i in range(n_jobs):
    # Ensure file exists 
    curr_path = path_prefix + str(i) + ".pt"
    if not os.path.isfile(curr_path):
        print("File:", curr_path, "not found! Skipping outputs")
        continue

    graphs = torch.load(curr_path)
    # Handle first case, if ys is none, just set it equal to this graphs ys
    if ys == None:
        ys = [graph.y for graph in graphs]
        continue

    # Otherwise, if a graph.y contains a non -1 value, replace the value in ys
    ys_update = [graph.y for graph in graphs]
    for i in range(len(ys)):
        for j in range(len(ys[i])):
            if ys[i][j] == -1 and ys_update[i][j] != -1:
                ys[i][j] = ys_update[i][j]

# Take first set of graphs and update with merged ys, save that
graphs = torch.load(path_prefix + "0.pt")
for i in range(len(graphs)):
    graphs[i].y = ys[i]
print("Saving combined outputs to {path_prefix}Merged.pt")
torch.save(graphs, path_prefix + "Merged.pt")
