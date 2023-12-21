import cv2
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from spatial_sensor import Spatial2D
import random
from time import process_time
from tqdm import tqdm
import utils

class LIF:
    def __init__(self, neuron_id: str, lif_init: str = "default", trim_lim: int = 10, verbose_log: bool = False) -> None:
        self.neuron_id = neuron_id
        if lif_init == "default":
            self.tau_m = np.float16(1.5)  # Membrane time constant
            self.V_reset = np.float16(-75.0)  # Reset voltage
            self.V_threshold = np.float16(-55.0)  # Spike threshold
        elif lif_init == "random":
            self.tau_m = np.float16(random.uniform(0.000001, 0.999999))  # Membrane time constant
            self.V_reset = np.float16(random.uniform(-80.0, -70.0))  # Reset voltage
            self.V_threshold = np.float16(random.uniform(-55.0, -20.0))  # Spike threshold

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim

        self.verbose_log = verbose_log
        self.full_spike_log = list()

    # Define a function to update the LIF neuron's state
    def update(self, current_input: np.float16 = np.float16(0)):
        if len(self.spike_log) >= self.trim_lim:
            del self.spike_log[0]

        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            #delta_V = (current_input - self.V_reset) / self.tau_m
            #self.V.append(self.V_reset + delta_V)
            self.V.append(self.V_reset)
        else:
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V)

        if self.V[-1] >= self.V_threshold:
            if self.spike_bool:
                self.V[-1] = self.V_reset
                self.spike_bool = False
            else:
                self.V[-1] = self.V_threshold
                self.spike_log.append(1)
                self.spike_bool = True
            if self.verbose_log:
                self.full_spike_log.append(1)
        else:
            self.spike_log.append(0)
            self.spike_bool = False
            if self.verbose_log:
                self.full_spike_log.append(0)

class WeightMatrix:
    def __init__(self, neuron_keys: int, w_init: str = "default"):
        self.neuron_keys = neuron_keys
        self.n_neurons = len(neuron_keys)
        if w_init == "default":
            self.matrix = np.zeros(shape=(self.n_neurons, self.n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.rand(self.n_neurons, self.n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        self.matrix = pd.DataFrame(self.matrix, columns=neuron_keys, index=neuron_keys)
    def PrintMatrix(self):
        print(self.matrix)
        print(self.matrix.shape)
        print(self.matrix.columns)
        #raise

class Network:
    def __init__(self,
                 n_neurons: int,
                 audio_input: bool,
                 image_input: bool,
                 lif_init: str = "default",
                 w_init: str = "default",
                 hist_lim: int = 10,
                 verbose_logging:bool = False) -> None:

        self.stdp_lr        = np.float16(0.01)
        self.hebbian_lr     = np.float16(0.001)
        self.weight_decay   = np.float16(0.0005)

        self.n_neurons = n_neurons
        self.LIFNeurons = dict()
        self.audio_input = audio_input
        self.image_input = image_input

        self.w_init = w_init
        self.weight_log = []
        
        self.signal_cache = dict()
        self.clear_signal_cache = False

        self.hist_lim = hist_lim
        self.lif_init = lif_init
        
        self.verbose_logging = verbose_logging

    def InitNetwork(self):
        self.Spatial2D = Spatial2D(r = 32)
        for i in range(self.Spatial2D.total):
            self.LIFNeurons[f"Input {i}"] = LIF(i,
                trim_lim=self.hist_lim,
                lif_init=self.lif_init,
                verbose_log=self.verbose_logging)
        for i in range(self.n_neurons):
            self.LIFNeurons[str(i)] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.verbose_logging)
        self.n_neurons += self.Spatial2D.total

        self.neuron_keys = list(self.LIFNeurons.keys())

        self.weightsclass = WeightMatrix(self.neuron_keys, self.w_init)
        #self.weightsclass.PrintMatrix()

        self.weight_matrix = self.weightsclass.matrix

    def Decay(self, n1: str, n2: str, factor: float):
        factor = np.float16(factor)
        old_weight = self.weight_matrix[n1][n2]
        self.weight_matrix[n1][n2] -= factor * old_weight

    def Hebbian(self, n1: str, n2: str):
        pre_ss = self.LIFNeurons[n1].spike_log
        post_ss = self.LIFNeurons[n2].spike_log
        if len(pre_ss) == len(post_ss) and len(pre_ss) >= 1:
            correlation_term = np.dot(pre_ss, post_ss)
            new_weight = self.weight_matrix[n1][n2] + np.dot(self.hebbian_lr, correlation_term)
            if correlation_term <= 0.49:
                self.Decay(n1, n2, factor=self.weight_decay)
            else:
                self.weight_matrix[n1][n2] += new_weight

    def PrepSignals(self, fired_list:list):
        # Attemp signal propagation on every step
        # If there were no fired neurons, skip.
        cache_dict = dict()
        for fired_k in fired_list:
            for other_k in self.neuron_keys:
                if fired_k != other_k:
                    weight = self.weight_matrix[fired_k][other_k]
                    signal = self.LIFNeurons[fired_k].V_threshold
                    cache_dict[str(other_k)] = (signal*weight)

        return cache_dict

    def step(self, input_current = np.float16(0.0000), input_neuron:str = "0", fired_input_keys = []):
        if input_current != np.float16(0.0000):
            input_current = input_current / np.pi

        fired_neuron_keys = fired_input_keys
        signal_keys = list(self.signal_cache.keys())
        
        signal_keys = [s_k for s_k in signal_keys if "Input" not in s_k]
        filtered_keys = [n_k for n_k in self.neuron_keys if "Input" not in n_k]

        # We need to skip audio sensors as it is always ran first
        if len(signal_keys) > 0:
            for receiver_neuron in signal_keys:
                r_k = receiver_neuron
                recieved_signal = self.signal_cache[str(r_k)]
                neu = self.LIFNeurons[r_k]
                if str(r_k) == str(input_neuron):
                    neu.update(input_current+recieved_signal)
                else:
                    neu.update(np.float16(recieved_signal))

                if neu.spike_bool:
                    fired_neuron_keys.append(r_k)
                filtered_keys.remove(r_k)
                #print("\t", len(neu.V), r_k, recieved_signal)


        for k in filtered_keys:
            neu = self.LIFNeurons[k]
            if str(k) == str(input_neuron):
                neu.update(input_current)
            else:
                neu.update(np.float16(0))

            if neu.spike_bool:
                fired_neuron_keys.append(k)
            #print("\t", len(neu.V), k, input_current)

        if len(fired_neuron_keys) >= 1:
            self.signal_cache = self.PrepSignals(fired_neuron_keys)

        # Do Global Weight Update
        for k1 in self.neuron_keys:
            for k2 in self.neuron_keys:
                if k1 != k2:
                    self.Hebbian(k1, k2)

        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))

        # Copy weight matrix to a logger
        self.weight_log.append(np.copy(self.weight_matrix.to_numpy()))
        del fired_neuron_keys
        del filtered_keys

    def step_spatial_input(self, current_vector: np.ndarray):
        fired_cache = []
        update_cache = {}
        for input_id in range(self.Spatial2D.total):
            pixel_index = input_id
            pixel = current_vector[pixel_index]

            input_id = f"Input {input_id}"
            #update_cache[input_id] = self.LIFNeurons[input_id]
            #neu = update_cache[input_id]
            neu = self.LIFNeurons[input_id]
            neu.update(pixel)

            if neu.spike_bool:
                fired_cache.append(input_id)
                print("BANG")
            print("\t", len(neu.V), input_id, pixel, neu.V[-1])
        #print(update_cache.values())
        return (fired_cache, update_cache)

    def run_spatial(self, ticks):
        # SAMPLE
        image = cv2.imread("./sample.png", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32), cv2.INTER_AREA)
        current_data = utils.to_current(image)
        current_vector = utils.to_vector(current_data)
        input_data = np.float16(-55.0)

        # Run initial step!
        snn.step(input_current = input_data, input_neuron= "",
                    fired_input_keys=[])
        fired_cache, update_cache = snn.step_spatial_input(current_vector)
        #for n_k in self.neuron_keys:
        #    print(f"{n_k}\t{self.LIFNeurons[n_k].V}")
        #raise
        #for i in tqdm(range(ticks)):
        for i in range(ticks):
            print(i)
            with multiprocessing.Pool(processes=4) as pool:
                for step_return in pool.imap_unordered(snn.step_spatial_input, [current_vector, ]):
                    fired_cache = step_return[0]
                    update_cache = step_return[1]
                    #print(update_cache)
            #p = Pool(6)
            #for 
            #fired_cache = snn.step_spatial_input(current_vector)
            snn.step(input_current = input_data, input_neuron= "",
                     fired_input_keys=fired_cache)

    def SaveWeightTables(self, mode = "npy"):
        if mode == "npy":
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./weight_logs.npy", self.weight_log)
        else:
            print("No save format was defined, saving in .npy format!")
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./weight_logs.npy", self.weight_log)

    def SaveNeuronPotentials(self):
        format_cache = []
        for k in list(self.neuron_keys):
            #print(self.LIFNeurons[k].V)
            format_cache.append(np.asarray(self.LIFNeurons[k].V))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_V_logs.npy", format_cache)
    
    def SaveNeuronSpikes(self):
        # Check if the neurons are logged verbosely
        len(self.LIFNeurons["0"].full_spike_log)
        if len(self.LIFNeurons["0"].full_spike_log) < 1:
            e = Exception("Neurons were not initialized with 'verbose_log' to 'True' !")
            raise e
        format_cache = []
        for k in list(self.neuron_keys):
            #print(len(self.LIFNeurons[k].full_spike_log))
            format_cache.append(np.asarray(self.LIFNeurons[k].full_spike_log))
        format_cache = np.asarray(format_cache)
        np.save("./neuron_spike_logs.npy", format_cache)


if __name__ == "__main__":
    snn = Network(
        n_neurons = 26,
        lif_init = "default",
        w_init="random",
        hist_lim=17,
        verbose_logging = True,
        audio_input=False,
        image_input=True)
    snn.InitNetwork()
    #print(snn.neuron_keys)
    snn.run_spatial(1)
    #snn.SaveWeightTables()
    for i in snn.neuron_keys:
        neu = snn.LIFNeurons[f"{i}"]
        #print(i)
        #print(neu.V)
    #snn.SaveNeuronSpikes()
    #snn.SaveNeuronPotentials()
    