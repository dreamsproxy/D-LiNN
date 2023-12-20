import numpy as np
import plotly.graph_objects as go 
import plotly.express as px 
import pandas as pd
import seed_generators
from seed_generators import random_seed
import lsm
from scipy.spatial import distance

# Assume we want to simulate 256 neurons in a 256^3 environment.
nodes = 4
env_size = (256, 256, 255)
sim_space = np.zeros(env_size)
# Generate a weight matrix with neuron IDs assigned.
node_keys = [str(i) for i in range(nodes)]
weight_class = lsm.WeightMatrix(node_keys, w_init="zeros")
weight_matrix = weight_class.matrix

# Using coordinate generator fuunctions from seed_generators
# We will calculate the distance between each LIF node
node_coords = random_seed(nodes, max_size=256)

# Cross Calc
for i1, start_coord in enumerate(node_coords):
    for i2, dest_coord in enumerate(node_coords):
        if i1 != i2:
            dst = distance.euclidean(start_coord, dest_coord)
            weight_matrix[str(i1)][str(i2)] = dst

network = lsm.Network(
    nodes,
    lif_init="default",
    w_init="zeros",
    hist_lim=17,
    verbose_logging=True)
network.InitNetwork()
network.weight_matrix = weight_matrix

sim_ticks = 100
input_data = np.float16(-55.0)
stream_points = seed_generators.fibonacci(sim_ticks)

from tqdm import tqdm
for i in tqdm(range(sim_ticks)):
    if i in stream_points:
        network.step(input_current = input_data, input_neuron= "0")
    else:
        network.step(input_current = 0.0, input_neuron= "0")

network.SaveWeightTables()