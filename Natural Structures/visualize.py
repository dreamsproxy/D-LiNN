import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def WeightMatrix(path = "./logs/weight_logs.npy", trim_last: int = 0):
    weight_logs = []
    weight_matrix = np.load(path)
    with open("./logs/ids.txt", "r") as infile:
        neuron_ids = infile.read().split(",")
    if trim_last >= 1:
        weight_matrix = weight_matrix[trim_last:-1]
    for tick_slice in weight_matrix:
        df_tick_slice = pd.DataFrame(tick_slice, columns=neuron_ids)
        df_tick_slice.insert(0, "ID", neuron_ids)
        df_tick_slice.set_index(['ID'], inplace= True)
        for i in neuron_ids:
            if "Alpha" in i:
                df_tick_slice.drop([f"{i}"], inplace=True)
        df_tick_slice.transpose(copy=False)
        for i in neuron_ids:
            if "Alpha" in i:
                df_tick_slice.drop([f"{i}"], axis=1, inplace=True)
        weight_logs.append(df_tick_slice)
    frames = [
        go.Frame(
            data=go.Heatmap(
                z=frame.values,
                x=frame.columns,
                y=frame.index,
                colorscale="viridis"),
            name=i)
        for i, frame in enumerate(weight_logs)
    ]

    fig = go.Figure(data=frames[0].data, frames=frames).update_layout(
        updatemenus=[
            {
                "buttons": [{"args": [None, {"frame": {"duration": 5, "redraw": True}}],
                            "label": "Play", "method": "animate",},
                            {"args": [[None],{"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate", "transition": {"duration": 0},},],
                            "label": "Pause", "method": "animate",},],
                "type": "buttons",
            }
        ],
        # iterate over frames to generate steps... NB frame name...
        sliders=[{"steps": [{"args": [[f.name],{"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",},],
                            "label": f.name, "method": "animate",}
                            for f in frames],}],
        height=1000,
        width=1000,
        yaxis={"title": "Pre-Synaptic IDs", "tickangle":0},
        xaxis={"title": "Post-Synaptic IDs", "side": "top"},
        title_x=0.5,

    )
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.show()

def NeuronSpikes(file_type = ".npy"):
    import plotly.express as px
    spikes = np.load("./logs/neuron_spike_logs.npy")
    with open("./logs/ids.txt", "r") as infile:
        ids = infile.read().split(",")

    # Convert to dataframe
    spikes_df = pd.DataFrame(np.swapaxes(spikes, 0, 1), columns=ids)
    spikes_df.set_index(ids)

    fig = px.line(spikes_df, title="Neuron Spike")
    fig.show()

def NeuronPotentials():
    import plotly.express as px
    potentials = np.load("./logs/neuron_V_logs.npy")
    with open("./logs/ids.txt", "r") as infile:
        ids = infile.read().split(",")

    # Convert to dataframe
    potentials_df = pd.DataFrame(np.swapaxes(potentials, 0, 1), columns=ids)
    potentials_df.set_index(ids)
    print(potentials_df.head())

    fig = px.line(potentials_df, title="Neuron Potentials")
    fig.update_layout(
        yaxis={"title": "Voltage", "tickangle":90},
        xaxis={"title": "Ticks", "side": "bottom"},
        title_x=0.5,

    )
    fig.show()

def output():
    array = np.load("./output.npy")
    print(array)
    plt.imshow(array, cmap="gray")
    plt.show()
if __name__ == "__main__":
    WeightMatrix()
    #NeuronPotentials()
    #NeuronSpikes()
    
    #output()
