import numpy as np
from vispy import scene, app
import pandas as pd
import json

def load_weights(path = "./weight_logs.npy"):
    weights_npy = np.load(path)
    weight_logs = []
    with open("keys.txt", "r") as infile:
        col_keys = infile.readlines()

    col_keys = [i.replace("\n", "") for i in col_keys]
    for tick_slice in weights_npy:
        df_tick_slice = pd.DataFrame(tick_slice, columns=col_keys)
        weight_logs.append(df_tick_slice)
    return weight_logs

def to_dict(weight_logs):
    latest = weight_logs[-1]
    ids = list(latest.columns)
    
    weights_dict = dict()
    for k in ids:
        temp_dict = latest[k].to_dict()
        for idx, nest_k in enumerate(list(temp_dict.keys())):
            temp_dict[ids[idx]] = temp_dict[nest_k]
            del temp_dict[nest_k]
        temp_dict = {str(ke):np.float16(v) for ke, v in temp_dict.items()}
        try:
            del temp_dict[str(k)]
        except:
            continue
        weights_dict[str(k)] = temp_dict
    return weights_dict

def load_coords(path = "coordinates.json"):
    with open(path, "r") as infile:
        temp_dict = json.load(infile)
    coordinates = dict()
    for id in list(temp_dict.keys()):
        formated = [np.float16(ax) for ax in temp_dict[id].split(", ")]
        coordinates[id] = tuple(formated)
    
    return coordinates

def adjust_coordinates(coordinates, nested_weights):
    nodes = list(coordinates.keys())
    node_positions = np.array(list(coordinates.values()))
    for source, targets in nested_weights.items():
        if "Input " not in source:
            source_index = nodes.index(source)
            for target, weight in targets.items():
                target_index = nodes.index(target)
                node_positions[source_index] += (node_positions[target_index] - node_positions[source_index]) * np.float16(weight / np.float16(2))

    return dict(zip(nodes, node_positions))

coordinates = load_coords()
nested_weights = load_weights()
nested_weights = to_dict(nested_weights)

coordinates = adjust_coordinates(coordinates, nested_weights)

# Extract node positions and edges
nodes = list(coordinates.keys())

node_positions = np.array(list(coordinates.values()))

edges = []
#print(list(nested_weights.items())[0][0])

for source, targets in nested_weights.items():
    for target, weight in targets.items():
        if weight >= 0.33:
            edges.append((nodes.index(source), nodes.index(target)))
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1280, 800), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'

# Set the center of the camera to a specific point
center_point = np.mean(node_positions, axis=0)  # Set center to the mean of node positions
view.camera.center = center_point

# Create nodes
scatter = scene.visuals.Markers()
scatter.set_data(node_positions, edge_color='white', face_color='white', size=5)
view.add(scatter)

# Create edges
lines = scene.visuals.Line(pos=np.array(node_positions), color='blue')
lines.set_data(node_positions[edges], width=0.1)
#lines.set_data(node_positions[edges, 0], node_positions[edges, 1])  # Adjusted line to properly extract data
view.add(lines)

if __name__ == '__main__':
    app.run()