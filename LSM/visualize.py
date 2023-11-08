import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt

file_list = glob("./weight_logs/*WM.csv")

"""
preproccess = []

for csv_file in file_list:
    temp = pd.read_csv(csv_file).transpose()
    temp = temp.drop(temp.index[0])
    preproccess.append(temp)

weight_log = []
for frame in preproccess:
    weight_log.append(frame["0"])

#weight_log = pd.concat(weight_log, axis = 1, ignore_index= True)
#weight_log = weight_log.transpose()

#weight_log = weight_log.drop(weight_log.index[0])
#print(weight_log.head())

#sns.lineplot(weight_log)
#plt.show()"""

weight_logs = []
weight_matrix = np.load("./weight_logs.npy")
#print(weight_matrix.shape)
#print(weight_matrix[0])
for tick_slice in weight_matrix:
    df_tick_slice = pd.DataFrame(tick_slice, columns=[str(x) for x in range(tick_slice.shape[0])])
    weight_logs.append(df_tick_slice)

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# generate the frames. NB name
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

def PlotWeightMatrix(self):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import animation
    num_frames = len(self.weight_log)
    print(num_frames)
    
    # Set the background color for the plot
    sns.set(rc={'axes.facecolor':'#002439', 'figure.facecolor':'#002439'})

    # extract initial frame
    init_frame = self.weight_log[0]
    self.weight_log.pop(0)
    def init():
        # Initialize the heatmap (use the first frame as the initial state)
        heatmap = sns.heatmap(
            init_frame,
            square=True,
            cmap="mako",
            annot=True,
            annot_kws={'size': 8},
            fmt = ".2f"
            )
        heatmap.invert_yaxis()
        heatmap.set_xticklabels(heatmap.get_xticklabels(), color="white")
        heatmap.set_yticklabels(heatmap.get_yticklabels(), color="white")
        heatmap.set_title("Weight Matrix").set_color("white")
        heatmap.title.set_fontsize(20)

    fig = plt.figure()

    def animate(i):
        data = self.weight_log[i]
        sns.heatmap(data, square=True, cmap="mako", annot=True, annot_kws={'size': 8}, fmt = ".2f", cbar=False)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames-1, repeat=True)
    save_prog = tqdm(total = num_frames)
    
    anim.save("mat.gif", fps=2, progress_callback=save_prog.update(1))


def SaveWeightFrames(self):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(rc={'axes.facecolor':'#002439', 'figure.facecolor':'#002439'})
    for step, frame in enumerate(self.weight_log):
        plt.clf()
        fig = sns.heatmap(frame, square=True, cmap="deep", annot=True, annot_kws={'size': 8})
        fig.invert_yaxis()
        fig.set_xticklabels(fig.get_xticklabels(), color="white")
        fig.set_yticklabels(fig.get_yticklabels(), color="white")
        
        fig.set_title("Weight Matrix").set_color("white")
        fig.title.set_fontsize(20)
        fig.figure.savefig(f"{step}.png", dpi = 1200)
        plt.close()  # Close the figure to release resources

def PrintNetworkV(self):
    neuron_keys = list(self.LIFNeurons.keys())
    for i in neuron_keys:
        print(self.LIFNeurons[i].V)

def PlotNetworkV(self):
    import plotly.graph_objects as go
    
    fig = go.Figure()
    for key, data in self.LIFNeurons.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data.V))),
                y=data.V, mode='lines',
                name=key))
    fig.update_layout(title='Membrane Potential Log',
                    xaxis_title='Ticks',
                    yaxis_title='Voltage')
    fig.show()