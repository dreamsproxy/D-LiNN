import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

def random_seed(neuron_count: int = 256, max_size: int = 256):
    #Generate random coordinates within (size, size, size)
    x_coords = np.vstack(np.random.randint(0, high=max_size, size=(neuron_count)))
    y_coords = np.vstack(np.random.randint(0, high=max_size, size=(neuron_count)))
    z_coords = np.vstack(np.random.randint(0, high=max_size, size=(neuron_count)))
    neuron_coords = np.hstack((x_coords, y_coords, z_coords))

    return neuron_coords

def fibonacci(neuron_count: int = 256, size: int = 0):
    a = 0
    b = 1
    cache = []
    for i in range(1, neuron_count):
        c = a + b
        a = b
        b = c
        cache.append(b)
    cache = np.array(cache)
    return cache
    #relative_locs = minmax_scale(cache, feature_range=(0.0, 1.0))
    #if size > 0:
    #    relative_locs *= size
    #return relative_locs
