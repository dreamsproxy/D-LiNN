import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class AdaptiveLIFNeuron:
    def __init__(self, ID, threshold=1.0, tau_m=10.0, tau_s=1.0, tau_a=100.0, n_type = "core"):
        self.ID = str(ID)
        self.threshold = threshold
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.tau_a = tau_a
        self.n_type = n_type
        self.membrane_potential = 0.0
        self.adaptation = 0.0

    def integrate_and_fire(self, inputs):
        self.membrane_potential += (inputs - self.membrane_potential) / self.tau_m
        self.adaptation += (-self.adaptation) / self.tau_a
        self.membrane_potential -= self.adaptation
        if self.membrane_potential >= self.threshold: 
            self.membrane_potential = 0.0
            self.adaptation += 0.1  # Increment adaptation after firing
            return self.threshold
        else:
            pass

class LSM:
    def __init__(self, num_neurons):
        self.neuron_names = [f"N-{num}" for num in range(num_neurons)]
        self.latest_dict = None
        self.weight_table = []
        self.neuron_table = [AdaptiveLIFNeuron(ID = f"N-{num}") for num in self.neuron_names]
        self.local_DHT = None
        self.latest_weights = []
        self.initaliized = False
        
    def InitWeights(self):
        initial_weights = []
        for iter in self.neuron_names:
            initial_weights.append([np.random.uniform(low=0.0, high=0.9) for _ in self.neuron_names])
        
        self.latest_dict = {key: id for key, id in zip(self.neuron_names, initial_weights)}
        df_weights = pd.DataFrame(self.latest_dict, columns=self.neuron_names, index=self.neuron_names)
        df_weights.to_csv("./network_logs/init.csv", index = True, columns = self.neuron_names)
        
        self.weight_table.append(df_weights)
        
        self.initaliized = True
    
    def UpdateDHT(self):
        for neuron in self.neuron_names:
            pass
        pass
    
    def GetHistoryDHT(self):
        return self.weight_table
    
    def GetInitDHT(self):
        return self.weight_table[0]
    
    def step(self):
        for neuron in self.neuron_table:
            if neuron.type == "core":
                # neighbor signal is currently for testing purposes
                neighbor_signal = 1.0
                neighbor_out = neuron.integrate_and_fire(inputs = neighbor_signal)
                for neighbor in self.weight_table:
                    neighbor_out
                    # Search all neighbors
                    # all neighbors recieve neightbor_out * weight
            elif neuron.type == "input":
                neuron.integrate_and_fire(inputs = input_signal)
#lsm = LSM(2, 1, 4)
lsm = LSM(4)
lsm.InitWeights()

initial_DHT = lsm.GetInitDHT()
print(initial_DHT)
#print(lsm.GetInitDHT)
#sns.heatmap(initial_DHT, cmap='coolwarm', annot=True, fmt=".1f")
sns.heatmap(initial_DHT, square=True, cmap="viridis")
plt.show()