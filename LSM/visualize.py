import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt

file_list = glob("./weight_logs/*WM.csv")

preproccess = []
for csv_file in file_list:
    temp = pd.read_csv(csv_file).transpose()
    temp = temp.drop(temp.index[0])

weight_log = []
for frame in preproccess:
    weight_log.append(frame["0"])

#weight_log = pd.concat(weight_log, axis = 1, ignore_index= True)
#weight_log = weight_log.transpose()

#weight_log = weight_log.drop(weight_log.index[0])
#print(weight_log.head())

#sns.lineplot(weight_log)
#plt.show()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
3

# generate the frames. NB name
frames = [
    go.Frame(data=go.Heatmap(z=preproccess.values, x=preproccess.columns, y=preproccess.index), name=i)
    for i, preproccess in enumerate(preproccess)
]

fig = go.Figure(data=frames[0].data, frames=frames).update_layout(
    updatemenus=[
        {
            "buttons": [{"args": [None, {"frame": {"duration": 500, "redraw": True}}],
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
    height=800,
    yaxis={"title": 'callers'},
    xaxis={"title": 'callees', "tickangle": 45, 'side': 'top'},
    title_x=0.5,

)

fig.show()