import numpy as np
import plotly.graph_objects as go 
import plotly.express as px 
import pandas as pd
import seed_generators
from seed_generators import random_seed, generate_grids
import lsm
from scipy.spatial import distance
import json
from tqdm import tqdm
import utils


# Assume we want to simulate 256 hidden neurons in a 256^3 environment.
nodes = 16
env_size = (256, 256, 256)
sim_space = np.zeros(env_size)

# Assume we have a 2D Plane of inputs, imagine a filter screen.
r=16
s=r*2
input_plane = generate_grids(256, 256, 600, s, r)
input_plane = np.reshape(input_plane, (input_plane.shape[0] * input_plane.shape[1], 3))
input_keys = [f"Input {i}" for i in range(len(input_plane))]

# Generate a weight matrix with neuron IDs assigned.
node_keys = [str(i) for i in range(nodes)]

node_keys = node_keys + input_keys
weight_class = lsm.WeightMatrix(node_keys, w_init="zeros")
weight_matrix = weight_class.matrix

# Using coordinate generator fuunctions from seed_generators
# We will calculate the distance between each LIF node
node_coords = random_seed(nodes, max_size=512)
node_coords = np.vstack((node_coords, input_plane))

node_coord_dict = dict()

# Cross Calc
for i1, start_coord in tqdm(enumerate(node_coords), total = len(node_coords)):
    node_coord_dict[node_keys[i1]] = str(tuple(start_coord)).replace("(", "").replace(")", "")
    for i2, dest_coord in enumerate(node_coords):
        if i1 != i2:
            dst = distance.euclidean(start_coord, dest_coord)
            weight_matrix[node_keys[i1]][node_keys[i2]] = dst

# Dump ID : Coord pair to json
with open("coordinates.json", "w") as outfile:
    json.dump(node_coord_dict, outfile)

network = lsm.Network(
    len(node_keys),
    lif_init="default",
    w_init="zeros",
    hist_lim=17,
    verbose_logging=True)
network.InitNetwork(custom_keys = node_keys)
network.weight_matrix = weight_matrix

sim_ticks = 2

input_data = np.float16(1.0)
stream_points = [i for i in range(0, sim_ticks, 3)]

for i in tqdm(range(sim_ticks)):
    if i in stream_points:
        network.step(input_current = input_data, input_neurons = input_plane)
    else:
        network.step(input_current = np.float16(0.0), input_neurons = input_plane)

network.SaveWeightTables()
network.SaveNeuronPotentials()
network.SaveNeuronSpikes()
with open("keys.txt", "w") as outfile:
    for i in node_keys:
        outfile.writelines(i)
        outfile.writelines("\n")
print(network.neuron_keys)
raise