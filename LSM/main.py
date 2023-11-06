import numpy as np
from tqdm import tqdm
import ffmpeg
class WeightMatrix:
    def __init__(self, n_neurons, w_init = "zeros"):
        self.n_neurons = n_neurons
        if w_init == "zeros" or w_init == None:
            self.matrix = np.zeros(shape=(n_neurons, n_neurons))
        elif w_init == "mid" or w_init == None:
            self.matrix = np.zeros(shape=(n_neurons, n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(n_neurons, n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        
        pairs = [[p, p] for p in range(n_neurons)]
        for p1, p2 in pairs:
            self.matrix[p1, p2] = 0
        
        # Each neuron will be assigned an X (row) number for referencing the matrix

        # Find a way to update the weights and how the neuron signals propagate to each other

    def PrintMatrix(self):
        print(self.matrix)
    
class LIF:
    def __init__(self, neuron_id: str, trim_lim: int = 10) -> None:
        self.neuron_id = neuron_id
        # Define simulation parameters
        self.dT = 0.1  # Time step
        self.tau_m = np.float16(10.0)  # Membrane time constant
        self.V_reset = np.float16(0.0)  # Reset voltage
        self.V_threshold = np.float16(1.0)  # Spike threshold

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim

    # Define a function to update the LIF neuron's state
    def update(self, current_input = np.float16(0)):
        if len(self.spike_log) >= self.trim_lim:
            del self.spike_log[0]
        else:
            pass
        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            delta_V = (current_input - np.float16(0.000)) / self.tau_m
            self.V.append(np.float16(0.000) + delta_V)
        else:
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            self.V[-1] = self.V_reset
            self.spike_log.append("1")
            self.spike_bool = True
        else:
            self.spike_log.append("0")
            self.spike_bool = False


class Network:
    def __init__(self, n_neurons, w_init = None, hist_lim = 10) -> None:
        self.n_neurons = n_neurons
        self.LIFNeurons = dict()
        self.weightsclass = WeightMatrix(n_neurons, w_init)
        self.weightmatrix = self.weightsclass.matrix
        self.wave_dict = dict()
        self.weight_log = []
        self.spike_mem = {}
        self.hist_lim = hist_lim

    def InitNetwork(self):
        for i in range(self.n_neurons):
            self.LIFNeurons[i] = LIF(i, trim_lim=self.hist_lim)

    def PrepPropagation(self, neuron_id):
        neuron_keys = list(self.LIFNeurons.keys())
        neighbor_weights = self.weightmatrix[neuron_id, :]
        del neuron_keys[neuron_id]

        return neuron_keys, neighbor_weights

    def Decay(self, n1: str, n2: str, factor: float):
        # TODO
        # Implement a method where decay factor is relative to
        # the tick (T)
        factor = np.float16(factor)
        old_weight = self.weightmatrix[n1, n2]
        self.weightmatrix[n1, n2] -= factor * old_weight

    def Hebbian(self, n1: str, n2: str):
        # Neurons that fire together,
        # Connects together.
        source_act = self.LIFNeurons[n1].V_threshold
        target_act = self.LIFNeurons[n2].V_threshold

        old_weight = self.weightmatrix[n1, n2]
        if old_weight < np.float16(1.000):
            new_weight = old_weight + (old_weight * np.float16(0.1))
            if new_weight >= np.float16(1.000):
                new_weight = np.float16(1.000)
                self.weightmatrix[n1, n2] = new_weight
            elif new_weight < np.float16(1.0):
                self.weightmatrix[n1, n2] = new_weight
        elif old_weight >= np.float16(1.000):
            new_weight = np.float16(1.000)
            self.weightmatrix[n1, n2] = new_weight

    def step(self, tick,  input_current = np.float16(0.000), input_neuron = 0):
        neuron_keys = list(self.LIFNeurons.keys())
        for k in neuron_keys:
            spike_collector = []
            neu = self.LIFNeurons[k]
            if neu.neuron_id == input_neuron:
                neu.update(input_current)
            else:
                neu.update(np.float16(0))

            # Collect IDs of all neurons that spiked.
            if neu.spike_bool:
                spike_collector.append(neu.neuron_id)

            # Do Global Weight Update
            for n1 in spike_collector:
                for n2 in spike_collector:
                    if n1 != n2:
                        self.Hebbian(n1, n2)
            
            # Decay is only called every 'hist_lim' ticks
            if tick % self.hist_lim:
                # Filter the neurons that did not fire.
                non_fire_neurons_list = [x for x in neuron_keys if x not in spike_collector]
                # Decay the neurons that did not fire.
                for non_fire_id in non_fire_neurons_list:
                    if "1" not in self.LIFNeurons[non_fire_id].spike_log:
                        for n2 in non_fire_neurons_list:
                            if non_fire_id != n2:
                                self.Decay(non_fire_id, n2, factor=0.001)
        #print(self.weightmatrix.shape)
        self.weight_log.append(np.copy(self.weightmatrix))

    def SaveWeightTables(self):
        import pandas as pd
        cols = list(self.LIFNeurons.keys())
        for tick, table in tqdm(enumerate(self.weight_log), total=len(self.weight_log)):
            frame = pd.DataFrame(table)
            frame.columns = cols
            frame.set_index(cols)
            frame.to_csv(f"./weight_logs/{tick} WM.csv")

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


if __name__ == "__main__":
    snn = Network(n_neurons = 10, w_init="random", hist_lim=20)
    snn.InitNetwork()
    #snn.step(np.float16(10.0), 0)
    #raise
    for i in tqdm(range(100)):
        if i % 10 == 0:
            snn.step(tick = i, input_current= np.float16(10.0), input_neuron= 0)
        if i % 30 == 0:
            snn.step(tick = i, input_current= np.float16(50.0), input_neuron= 0)
        else:
            snn.step(tick = i, input_current= np.float16(0.00), input_neuron= 0)
    #snn.PrintNetworkV()
    #print(snn.LIFNeurons[0].V)
    #snn.PlotNetworkV()
    #snn.SaveWeightFrames()
    snn.SaveWeightTables()
    #snn.PlotWeightMatrix()
    #print(snn.weight_log[0])
    #print()
    #print(snn.weight_log[-1])
    #print(len(snn.weight_log))
    #for key in list(snn.LIFNeurons.keys()):
    #    print(f"{key}\t{snn.LIFNeurons[key].spikes}")
    #print(snn.global_spike_log)
    #snn.GetNeuronWeights()
    #WeightMatrix.PrintMatrix()
