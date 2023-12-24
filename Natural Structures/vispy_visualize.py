import numpy as np
from vispy import scene, app
import pandas as pd
import json

def load_weights(path = "./weight_logs.npy"):
    weight_logs = []
    weight_matrix = np.load(path)
    with open("ids.txt", "r") as infile:
        neuron_ids = infile.read().split(",")

    for tick_slice in weight_matrix:
        df_tick_slice = pd.DataFrame(tick_slice, columns=neuron_ids)
        df_tick_slice.insert(0, "ID", neuron_ids)
        df_tick_slice.set_index(['ID'], inplace= True)
        df_tick_slice.transpose(copy=False)
        weight_logs.append(df_tick_slice)
    return weight_logs

def to_dict(weight_logs):
    latest = weight_logs[-1]
    ids = list(latest.columns)
    
    weights_dict = dict()
    for k in ids:
        temp_dict = latest[k].to_dict()
        
        for nest_k in list(temp_dict.keys()):
            if temp_dict[nest_k] == np.nan:
                del temp_dict[nest_k]
            if "Alpha " in nest_k and "Alpha " in k:
                del temp_dict[nest_k]
        temp_dict = {str(ke):np.float16(v) for ke, v in temp_dict.items() if v != np.nan}
        try:
            del temp_dict[str(k)]
        except:
            continue
        weights_dict[str(k)] = temp_dict
    #print(weights_dict)
    return weights_dict

def load_coords(path = "coordinates.json"):
    with open(path, "r") as infile:
        temp_dict = json.load(infile)
    coordinates = dict()
    for id in list(temp_dict.keys()):
        formated = [np.float16(ax) for ax in temp_dict[id].split(", ")]
        coordinates[id] = tuple(formated)
    for id in list(coordinates.keys()):
        if "Alpha" in id:
            c_temp = []
            for i, c in enumerate(coordinates[id]):
                if i == 2:
                    c *= 2
                c_temp.append(c*4)
            #print(type(coordinates[id]))
            coordinates[id] = tuple(c_temp)
    return coordinates

def adjust_coordinates(coordinates, nested_weights):
    nodes = list(coordinates.keys())
    node_positions = np.array(list(coordinates.values()))
    for source, targets in nested_weights.items():
        if "Input " not in source:
            source_index = nodes.index(source)
            for target, weight in targets.items():
                target_index = nodes.index(target)
                node_positions[source_index] += (node_positions[target_index] - node_positions[source_index]) * np.float16(np.float16(weight / np.float16(2)) / 2)

    return dict(zip(nodes, node_positions))

coordinates = load_coords()
nested_weights = load_weights()
#print(nested_weights)
nested_weights = to_dict(nested_weights)

coordinates = adjust_coordinates(coordinates, nested_weights)

# Extract node positions and edges
nodes = list(coordinates.keys())


node_positions = np.multiply(np.array(list(coordinates.values())), 1.5)

edges = []
for source, targets in nested_weights.items():
    for target, weight in targets.items():
        if weight >= 0.8:
            edges.append((nodes.index(source), nodes.index(target)))
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1280, 800), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'

# Set the center of the camera to a specific point
center_point = np.mean(node_positions, axis=0)  # Set center to the mean of node positions
view.camera.center = center_point

# Create nodes
scatter = scene.visuals.Markers()
scatter.set_data(node_positions, edge_color='white', face_color='white', size=12)
view.add(scatter)

# Create edges
lines = scene.visuals.Line(pos=np.array(node_positions), color='blue')
lines.set_data(node_positions[edges], width=0.001)
#lines.set_data(node_positions[edges, 0], node_positions[edges, 1])  # Adjusted line to properly extract data
view.add(lines)

if __name__ == '__main__':
    app.run()