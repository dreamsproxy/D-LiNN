import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale


def generate_grids(x_dim, y_dim, z_dim, s, r):
    x_c = x_dim//2
    y_c = y_dim//2
    x, y = np.mgrid[x_c-s:x_c+s:r, y_c-s:y_c+s:r]
    #x, y = np.mgrid[x_c-r:x_c+r:r, y_c-r:y_c+r:r]
    
    z = np.zeros((len(x), len(y)))
    z += z_dim
    coordinates = np.stack((x, y, z), axis=-1).astype(int)

    return coordinates

def random_coords(neuron_count: int = 256, max_size: int = 0,
                x_lim: int = 256, y_lim: int = 256, z_lim: int = 256):
    if max_size >= 1:
        x_lim = max_size
        y_lim = max_size
        z_lim = max_size

    #Generate random coordinates within (size, size, size)
    x_coords = np.vstack(np.random.randint(-512, high=x_lim, size=(neuron_count)))
    y_coords = np.vstack(np.random.randint(-512, high=y_lim, size=(neuron_count)))
    z_coords = np.vstack(np.random.randint(-256, high=z_lim, size=(neuron_count)))

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
