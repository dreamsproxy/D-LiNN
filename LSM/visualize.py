import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def WeightMatrix(path = "./weight_logs.npy"):
    if path[-3:] == "csv":
        # I FORGOT WHAT THIS IS
        file_list = glob("./weight_logs/*WM.csv")
        preproccess = []
        for csv_file in file_list:
            temp = pd.read_csv(csv_file).transpose()
            temp = temp.drop(temp.index[0])
            preproccess.append(temp)
        weight_log = []
        for frame in preproccess:
            weight_log.append(frame["0"])

        weight_log = pd.concat(weight_log, axis = 1, ignore_index= True)
        weight_log = weight_log.transpose()

        weight_log = weight_log.drop(weight_log.index[0])
        print(weight_log.head())

        sns.lineplot(weight_log)
        plt.show()

    else:
        weight_logs = []
        weight_matrix = np.load("./weight_logs.npy")
        for tick_slice in weight_matrix:
            df_tick_slice = pd.DataFrame(tick_slice, columns=[str(x) for x in range(tick_slice.shape[0])])
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
            yaxis={"title": 'Pre-Synaptic IDs', "tickangle":90},
            xaxis={"title": 'Post-Synaptic IDs', 'side': 'top'},
            title_x=0.5,

        )

        fig.show()
        raise

def NeuronSpikes(file_type = ".npy"):
    import plotly.express as px
    spikes = np.load("./neuron_spike_logs.npy")
    v_logs = np.load("./neuron_V_logs.npy")
    
    # Convert to dataframe
    spikes_df = pd.DataFrame(np.swapaxes(spikes, 0, 1), columns=[i for i in range(spikes.shape[0])])
    spikes_df.set_index([i for i in range(spikes.shape[0])])
    print(spikes_df.head())

    fig = px.line(spikes_df, title="Neuron Spike Chart")
    fig.show()

def NeuronPotentials():
    import plotly.express as px
    v_logs = np.load("./neuron_V_logs.npy")
    print(v_logs.shape)
    # Convert to dataframe
    v_logs_df = pd.DataFrame(np.swapaxes(v_logs, 0, 1), columns=[i for i in range(v_logs.shape[0])])
    v_logs_df.set_index([i for i in range(v_logs.shape[0])])
    print(v_logs_df.head())

    fig = px.line(v_logs_df, title="Neuron Spike Chart")
    fig.show()

if __name__ == "__main__":
    #NeuronSpikes()
    NeuronPotentials()