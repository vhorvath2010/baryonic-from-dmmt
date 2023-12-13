import ytree
import torch

print('loading data...')
arbor = ytree.load('/storage/home/hhive1/jw254/data/SG256-v3/rockstar_halos/trees/tree_0_0_0.dat') # Set your arbor here
trees = list(arbor[:])

subhalos = []

print('finding subhalos...')
for tree in trees:
    for node in tree['tree']:
        if node['pid'] > 0:
            node_info = (node['position'].value.tolist()[0:3], node['redshift'], node['id'])
            subhalos.append(node_info)

print(f'found {len(subhalos)} subhalos')
torch.save(subhalos, 'SG256_subhalos.pt')
