from turtle import position
import numpy as np
from tqdm import tqdm
import random
import multiprocessing
from multiprocessing import Array
import pandas as pd
from time import process_time
import Conv2D
import utils
from seed_generators import random_coords, generate_grids
import json
import pickle
import numpy as np
import mpmath
import random
rng = np.random.default_rng()
class LIF:
    def __init__(self, neuron_id: str, lif_init: str = "default", trim_lim: int = 10, refractory_period: int = 5, verbose_log: bool = False) -> None:
        self.neuron_id = neuron_id
        self.lif_init = lif_init
        # Define simulation parameters
        if lif_init == "default":
            self.tau_m = np.float16(1.5)  # Membrane time constant
            self.V_reset = np.float16(-75.0)  # Reset voltage
            self.threshold = np.float16(-55.0)  # Spike threshold
            self.refractory_period = refractory_period
        elif lif_init == "random":
            self.tau_m = np.float16(rng.uniform(1.0, 2.000))  # Membrane time constant
            self.V_reset = np.float16(rng.uniform(-80.0, -70.0))  # Reset voltage
            self.threshold = np.float16(rng.uniform(-50.0, -40.0))  # Spike threshold
            self.refractory_period = rng.integers(0, 5)

        self.threshold_factor = np.float16(1.0)
        self.remaining_refractory_time = 0

        self.dt = 0.0

        self.V = list()
        self.spike_log = list()
        self.spike_bool = False
        self.trim_lim = trim_lim
        self.verbose_log = verbose_log
        if self.verbose_log:
            self.full_spike_log = list()
        
        self.full_debug = []
        self.full_debug.append("INIT")

    def inihbit(self):
        x = len([i for i in self.spike_log if i >= self.threshold])
        y = x / self.trim_lim
        if y >= 0.666:
            self.threshold_factor = self.threshold_factor + np.float16(0.1)
            self.threshold *= self.threshold_factor
        elif y < 0.666 and self.threshold_factor > np.float16(1.100):
            self.threshold_factor = self.threshold_factor - np.float16(0.1)
            self.threshold *= self.threshold_factor
        else:
            pass

    # Define a function to update the LIF neuron's state
    def update(self, current_input: np.float16 = np.float16(0)):
        # If the voltage log is empty, assume it is at 0.0, then perform calculation
        if len(self.V) < 1:
            delta_V = (current_input - self.V_reset) / self.tau_m
            self.V.append(self.V_reset + delta_V * self.dt / self.tau_m)
        if self.spike_bool:
            self.spike_bool = False
            self.V.append(self.V_reset * 1.5)
        # Handle refractory period
        if self.remaining_refractory_time > 0:
            self.remaining_refractory_time -= 1
            delta_V = (current_input - self.V_reset) / self.tau_m
            self.V.append(self.V_reset + delta_V * self.dt / self.tau_m)
            self.spike_log.append(self.V_reset)
            self.spike_bool = False
            if self.verbose_log:
                self.full_spike_log.append(self.V_reset)
        else:
            # In refractory period
            self.remaining_refractory_time = self.refractory_period
            delta_V = (current_input - self.V[-1]) / self.tau_m
            self.V.append(self.V[-1] + delta_V * self.dt / self.tau_m)
            if self.V[-1] >= self.threshold:
                self.V[-1] = self.threshold
                
                self.spike_log.append(self.threshold)
                self.spike_bool = True
                if self.verbose_log:
                    self.full_spike_log.append(self.threshold)
                # Enter refractory period
                self.remaining_refractory_time = self.refractory_period
            else:
                self.spike_log.append(self.V_reset)
                self.spike_bool = False
                if self.verbose_log:
                    self.full_spike_log.append(self.V_reset)

        # Trim the spike log
        if len(self.spike_log) >= self.trim_lim:
            del self.spike_log[0]
        if self.lif_init == "default":
            self.inihbit()
        self.dt += 0.2

class WeightMatrix:
    def __init__(self, neuron_keys:list = [], w_init: str = "default"):
        self.neuron_keys = neuron_keys
        self.n_neurons = len(neuron_keys)
        if w_init == "default":
            self.matrix = np.zeros(shape=(self.n_neurons, self.n_neurons))+np.float16(0.5)
        elif w_init == "random":
            self.matrix = np.random.default_rng().uniform(np.float16(0.33), np.float16(0.99), (self.n_neurons, self.n_neurons))
            #self.matrix = np.random.rand(self.n_neurons, self.n_neurons)
        else:
            e = "\n\n\tWeight init only takes 'zeros' or 'random'!\n\tDefault is zero.\n"
            raise Exception(e)
        #self.matrix = pd.DataFrame(self.matrix, columns=neuron_keys, index=neuron_keys)
    def random_weight_purge(matrix):
        new_matrix = matrix
        x1 = matrix.shape[0]
        x2 = matrix.shape[0]//2
        skip_pairs = []
        group_a = rng.integers(0, x1, x2)
        group_b = rng.integers(0, x1, x2)
        for a in group_a:
            for b in group_b:
                if a != b:
                    new_matrix[a, b] == np.float16(0.0)
                    skip_pairs.append((a, b))
        return new_matrix, skip_pairs
    # Used for checking matrix (DEBUG)
    def PrintMatrix(self):
        print(self.matrix)
        print(self.matrix.shape)

global hebbian_lr
hebbian_lr     = np.float16(0.01)
global weight_penalty
weight_penalty = np.float16(0.50)

def adjust_weights_local(k1, k2, w, correlation_term):
    # Adjusts the weight, that is inside self.weight_matrix
    # Originally was a lambda func call, but more ops were needed
    term: np.float16 = np.multiply(hebbian_lr, correlation_term)
    new_w: np.float16 = w + term
    if new_w >= np.float16(0.90):
        new_w *= weight_penalty

    return (k1, k2, new_w)

class Network:
    def __init__(self,
                 n_neurons: int,
                 image_input: bool,
                 image_output: bool,
                 resolution: int,
                 lif_init: str = "random",
                 w_init: str = "default",
                 hist_lim: int = 17,
                 network_verbose_logging:bool = True,
                 neuron_verbose_logging:bool = False) -> None:

        self.n_neurons      = n_neurons

        # Where all neuron's keys and their LIF objects (KEY : child object)
        self.LIFNeurons     = dict()
        
        # Boolean, change to False and there will be no input image
        # If False, change input_neuronss: list in class method: "step()"
        # If False, remove "step_vision()" calls
        self.image_input    = image_input
        self.image_output   = image_output
        self.resolution     = resolution

        self.w_init         = w_init
        self.weight_log     = []
        
        # Essentially a backlog / to-do list for the network
        # If any neurons fired at step 0 completion:
        #       At next step, these neurons will be simulated first
        self.backlog   = dict()

        # Neuron parameters
        self.hist_lim       = hist_lim
        self.lif_init       = lif_init

        # Debug Params
        self.network_verbose_logging    = network_verbose_logging
        self.neuron_verbose_logging     = neuron_verbose_logging
        self.step_debug_log             = []

    def InitNetwork(self, n_inputs: int = 16):
        self.n_inputs = n_inputs
        self.resolution = self.resolution ** 2
        self.n_layers = 0
        self.key_map = dict()
        

        r = self.resolution
        s = 2
        for i in range(128):
            r = r // s
            if r == n_inputs:
                break
            else:
                self.n_layers += 1
        self.n_layers = int(self.n_layers - 1 ) // 2

        # Generating key : coordinate pairs for visualization
        coordinates = dict()
        coordinates_dump = dict()
        input_coords = generate_grids(2000, self.n_inputs, scale=1000)
        self.input_keys = list()
        for i in range(self.n_inputs):
            self.LIFNeurons[f"Input {i}"] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.neuron_verbose_logging)
            
            f = str(tuple(input_coords[i, :])).replace("(", "").replace(")", "")
            coordinates[f"Input {i}"] = input_coords[i, :]
            coordinates_dump[f"Input {i}"] = f
            self.input_keys.append(f"Input {i}")

        neuron_points = random_coords(self.n_neurons, x_lim=1000, y_lim=1000, z_lim=1500)
        for i in range(self.n_neurons):
            self.LIFNeurons[str(i)] = LIF(
                i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                verbose_log=self.neuron_verbose_logging)
            f = str(tuple(neuron_points[i])).replace("(", "").replace(")", "")
            coordinates[str(i)] = neuron_points[i]
            coordinates_dump[str(i)] = f

        if self.image_output:
            self.n_outputs = self.n_inputs
            output_coords = generate_grids(-500, self.n_outputs, scale=1000)
            for i in range(self.n_outputs):
                self.LIFNeurons[f"Output {i}"] = LIF(
                    i, trim_lim=self.hist_lim, lif_init = self.lif_init,
                    verbose_log=self.neuron_verbose_logging)
                
                f = str(tuple(output_coords[i, :])).replace("(", "").replace(")", "")
                coordinates[f"Output {i}"] = output_coords[i, :]
                coordinates_dump[f"Output {i}"] = f

        self.n_neurons += self.n_inputs
        if self.image_output:
            self.n_neurons += self.n_outputs
        self.neuron_keys = list(self.LIFNeurons.keys())
        self.skips = [i for i in range(self.n_outputs)]
        for idx, k in enumerate(self.neuron_keys):
            self.key_map[idx] = k

        # See WeightMatrix class:
        self.weightsclass = WeightMatrix(self.neuron_keys, self.w_init)
        self.weight_matrix = self.weightsclass.matrix

        return coordinates, coordinates_dump

    def convert_df_to_shared_array(self, df):
        array = df.to_numpy()
        shape = array.shape
        dtype = array.dtype

        shared_array = multiprocessing.Array(dtype, shape[0] * shape[1])
        shared_array_np = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
        np.copyto(shared_array_np, array)

        return shared_array

    def get_correlation_term(self, k1: str, k2: str):
        # Calculates the activity correlation between both
        # neurons.
        # Correlatiion term is the bias
        n1 = self.LIFNeurons[self.key_map[k1]].spike_log
        n2 = self.LIFNeurons[self.key_map[k2]].spike_log
        if len(n1) == len(n2) and len(n1) >= 2:
            correlation = np.corrcoef(n1, n2)[0, 1]
        else:
            correlation = 0.0
        return (k1, k2, correlation)

    def hebbian(self, k1, k2):
        k1_log = self.LIFNeurons[k1].spike_log
        if len(k1_log) > 1:
            pre_spike = k1_log[-2]
            post_spike = k1_log[-1]

            pre_neuron_index = k1_log.index(pre_spike)
            post_neuron_index = k1_log.index(post_spike)

            # Update the synaptic weight between pre and post neurons
            self.weight_matrix[pre_neuron_index, post_neuron_index] += hebbian_lr
    def BacklogHandler(self, fired_list:list):
        # The backlog handler
        # Called as a data formatter + handler
        # Handler stores to a backlog/todo list
        # Data is dumped to self.backlog
        # Data is then accessed at new step call
        cache_dict = dict()
        for f_k in fired_list:
            for other_k in list(self.key_map.keys()):
                if f_k not in self.skips and other_k not in self.skips:
                    if other_k != f_k:
                        weight = self.weight_matrix[f_k][other_k]
                        signal = self.LIFNeurons[self.key_map[f_k]].threshold
                        cache_dict[other_k] = (signal*weight)

        return cache_dict

    def step(self, step_number = 0, input_current = np.float16(0.0000), input_neurons:tuple = (), fired_input_keys = []):
        if self.network_verbose_logging:
            step_start = process_time()
            self.step_debug_log.append(f"\nStep {step_number} [START]")

        if input_current != np.float16(0.0000):
            if self.network_verbose_logging:
                self.step_debug_log.append("\n\tInput Current Detected!")
            input_current = input_current / np.pi
        else:
            if self.network_verbose_logging:
                self.step_debug_log.append("\tNo Input\n")

        fired_neuron_keys = []
        fired_neuron_keys += fired_input_keys
        signal_keys = list(self.backlog.keys())
        if self.image_input:
            signal_keys = [int(s_k) for s_k in signal_keys if "Input " not in self.key_map[s_k]]
            filtered_keys = [int(n_k) for n_k in list(self.key_map.keys()) if "Input " not in self.key_map[n_k]]
        #print(filtered_keys)
        self.step_debug_log.append(f"\n\tAll Neuron Keys:\n\t\t{self.neuron_keys}")
        if len(signal_keys) > 0:
            if self.network_verbose_logging:
                self.step_debug_log.append(f"\n\tBacklogs detected!")
                self.step_debug_log.append(f"\n\tBacklog Signal Keys: {signal_keys}")
                self.step_debug_log.append(f"\n\tBacklog [START]")
                backlog_start = process_time()
            print("Backlog:")
            for receiver_neuron in signal_keys:
                r_k = int(receiver_neuron)
                recieved_signal = self.backlog[r_k]
                #print(recieved_signal)
                neu = self.LIFNeurons[self.key_map[r_k]]
                if r_k in input_neurons:
                    neu.update(input_current+recieved_signal)
                    if self.network_verbose_logging:
                        self.step_debug_log.append(f"\n\t\tUpdate: input neuron {r_k}")
                else:
                    if self.network_verbose_logging:
                        self.step_debug_log.append(f"\n\t\tUpdate: neuron {r_k}")
                    neu.update(np.float16(recieved_signal))
                if neu.spike_bool:
                    fired_neuron_keys.append(r_k)
                if self.network_verbose_logging:
                    self.step_debug_log.append(f"\n\t\t\tRemoved: {r_k} from {filtered_keys}")
                filtered_keys.remove(r_k)
            if self.network_verbose_logging:
                backlog_end = process_time()
                self.step_debug_log.append(f"\n\tBacklog [ENDED]")
                self.step_debug_log.append(f"\n\tTOOK {backlog_end - backlog_start}")

        if self.network_verbose_logging:
            self.step_debug_log.append(f"\n\tNormal Step [START]")
            normal_start = process_time()
        #print(filtered_keys)
        for k in filtered_keys:
            if self.network_verbose_logging:
                self.step_debug_log.append(f"")
            neu = self.LIFNeurons[self.key_map[k]]
            if str(k) in input_neurons:
                if self.network_verbose_logging:
                    self.step_debug_log.append(f"\n\t\tUpdate: input neuron {k}")
                neu.update(input_current)
            else:
                if self.network_verbose_logging:
                    self.step_debug_log.append(f"\n\t\tUpdate: neuron {k}")
                neu.update(np.float16(-55.0))
            #print(k)

            if neu.spike_bool:
                fired_neuron_keys.append(k)

        #print(fired_neuron_keys)
        self.backlog.clear()
        if self.network_verbose_logging:
            normal_end = process_time()
            self.step_debug_log.append(f"\n\tNormal Step [ENDED]")
            self.step_debug_log.append(f"\n\t\tTOOK {normal_end - normal_start}")

        if len(fired_neuron_keys) >= 1:
            self.backlog = self.BacklogHandler(fired_neuron_keys)
            self.step_debug_log.append(f"\n\t\tThese Neurons Fired: {fired_neuron_keys}")
        else:
            if self.network_verbose_logging:
                self.step_debug_log.append(f"\n\t\tNO Neurons Fired")

        del fired_neuron_keys
        del filtered_keys
        if self.network_verbose_logging:
            step_end = process_time()
            self.step_debug_log.append(f"\nStep {step_number} [END]\n")
            self.step_debug_log.append(f"\nStep {step_number} TOOK {step_end - step_start}\n")
            step_total = step_end - step_start
            return step_total

    def GlobalOpt(self):
        if self.network_verbose_logging:
            self.step_debug_log.append(f"\nGLOBAL HEBBIAN WEIGHT OPT [START]")
            hebb_start = process_time()
        # Prepare data that can be passed to multiprocessing
        with multiprocessing.Pool(processes=6) as pool:
            # Example of using shared_weight_matrix in multiprocessing
            cache_result = pool.starmap_async(self.get_correlation_term, global_weight_pairs)
            result_args = cache_result.get()
            update_args = []
            for a in tqdm(result_args):
                k1, k2, c_term = a
                if k1 == k2:
                    print(k1, k2)
                w = self.weight_matrix[k1, k2]
                update_args.append((k1, k2, w, c_term))

            async_results = pool.starmap_async(adjust_weights_local, update_args)
            update_results = async_results.get()

        # Convert letters to array indices
        row_indices = np.array([row for row, _, _ in update_results])
        col_indices = np.array([col for _, col, _ in update_results])
        # Extract values from tuple_list
        values = np.array([value for _, _, value in update_results], dtype=np.float16)
        # Update the array in a vectorized manner
        self.weight_matrix[row_indices, col_indices] = values
        """
        for n in tqdm(update_results):
            k1, k2, new_w = n
            self.weight_matrix[k1, k2] = new_w
        """
        if self.network_verbose_logging:
            hebb_end = process_time()
            self.step_debug_log.append(f"\nGLOBAL HEBBIAN WEIGHT OPT [ENDED]")
            self.step_debug_log.append(f"\nWeight Normalization [START]")
            norm_start = process_time()

        # Normalize the weights to prevent uncontrolled growth
        self.weight_matrix /= np.max(np.abs(self.weight_matrix))
        # Scale the weights to keep it between 0.0 and 1.0
        self.weight_matrix = (self.weight_matrix-np.min(self.weight_matrix))/(np.max(self.weight_matrix)-np.min(self.weight_matrix))
        if self.network_verbose_logging:
            norm_end = process_time()
            if self.network_verbose_logging:
                self.step_debug_log.append(f"\nWeight Normalization [ENDED]")
                self.step_debug_log.append(f"\n\tTOOK {norm_end - norm_start}")

        # Copy weight matrix to a logger
        self.weight_log.append(np.copy(self.weight_matrix))

        if self.network_verbose_logging:
            self.step_debug_log.append(f"\n\t\tTOOK {hebb_end - hebb_start}")
            hebb_total = hebb_end - hebb_start
            return hebb_total

    def step_vision(self, current_vector: list):
        if self.network_verbose_logging:
            self.step_debug_log.append(f"\nVision Step [START]")
            vis_step_start = process_time()
        fired_cache = []
        for i in self.key_map.keys():
            str_id = self.key_map[i]
            if "Input " in str_id:
                neu = self.LIFNeurons[str_id]
                pixel = current_vector[i]
                neu.update(pixel)
                if neu.spike_bool:
                    fired_cache.append(i)
            if self.network_verbose_logging:
                self.step_debug_log.append(f"\nVision Step [END]")
                vis_step_end = process_time()
                self.step_debug_log.append(f"\n\tTOOK: {vis_step_end - vis_step_start}")
        
        return fired_cache

    def RunVision(self, ticks):
        import cv2
        from glob import glob
        paths = [f for f in glob("./SAMPLES/**")]
        images = []
        # Load imgs
        paths = glob("./samples/**")
        images = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            layers = dict()
            for i in range(self.n_layers):
                layers[str(i)] =  Conv2D.Conv2D(resolution = self.resolution, kernel_size = 3, n_strides=2)
            output_cache = []
            for i, k in enumerate(list(layers.keys())):
                if i == 0:
                    output_cache = layers[k].Call(img)
                else:
                    output_cache = layers[k].Call(output_cache)
            print(output_cache.shape)
            raise

            img = layers[str(self.n_layers-1)].Get(True)
            img = utils.to_vector(utils.to_current(img))
            images.append(img)
        counter = 0
        time_log = []
        for i in range(ticks):
            if i == 0:
                if counter >=  len(images):
                    counter = 0
                input_vector = images[counter]
            else:
                input_vector = np.zeros(shape=images[0].shape)
            print(f"\nStep {i}")
            input_data = np.float16(-55.0)
            fired = self.step_vision(input_vector)
            if self.network_verbose_logging:
                step_time = self.step(i, input_current = input_data, fired_input_keys=fired)
                hebb_time = self.GlobalOpt()
                step_total = step_time + hebb_time
                time_log.append(step_total)
                self.step_debug_log.append(f"\nTook: {step_total} seconds")
                print(f"Took: {step_total} seconds\n")
            else:
                self.step(i, input_current = input_data, fired_input_keys=fired)
                self.GlobalOpt()
            counter+= 1
        if self.network_verbose_logging:
            print(f"Average time taken: {np.mean(time_log)} s per step")

    def SaveWeightTables(self, mode = "npy"):
        if mode == "npy":
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./logs/weight_logs.npy", self.weight_log)
        else:
            print("No save format was defined, saving in .npy format!")
            total_ticks = len(self.weight_log)
            self.weight_log = np.asarray(self.weight_log)
            np.reshape(self.weight_log, (total_ticks, self.n_neurons, self.n_neurons))
            np.save("./logsweight_logs.npy", self.weight_log)

    def SaveNeuronPotentials(self):
        format_cache = []
        for k in list(self.neuron_keys):
            format_cache.append(np.asarray(self.LIFNeurons[k].V))
        format_cache = np.asarray(format_cache)
        np.save("./logs/neuron_V_logs.npy", format_cache)

    def SaveNeuronSpikes(self):
        # Check if the neurons are logged verbosely
        if len(self.LIFNeurons["0"].full_spike_log) < 1:
            e = Exception("Neurons were not initialized with 'verbose_log' to 'True' !")
            raise e
        format_cache = []
        for k in list(self.neuron_keys):
            format_cache.append(np.asarray(self.LIFNeurons[self.key_map[k]].full_spike_log))
        print(format_cache)
        format_cache = np.asarray(format_cache)
        np.save("./logs/neuron_spike_logs.npy", format_cache)

def EuclidianDistance(arg):
    k1, k2, c1, c2 = arg
    dist = [(a - b)**2 for a, b in zip(c1, c2)]
    dist = np.sqrt(sum(dist))
    return dist, k1, k2

if __name__ == "__main__":
    global key_pairs
    key_pairs = []
    global global_weight_pairs
    global_weight_pairs = []
    snn = Network(
        n_neurons = 256,
        image_input = True,
        image_output = True,
        resolution = 256,
        lif_init = "random",
        w_init = "default",
        hist_lim = 17,
        network_verbose_logging = True,
        neuron_verbose_logging = False)

    coords_dict, coords_dump = snn.InitNetwork(n_inputs=8)
    # Dump ID : Coord pair to json
    with open("./logs/coordinates.json", "w") as outfile:
        json.dump(coords_dump, outfile)

    print("Generating global neuron pairs:")
    # In-Out neurons skip
    ignore = ["Input ", "Output"]
    for k1 in snn.key_map.keys():
        for k2 in snn.key_map.keys():
            if k2 != k1:
                k1_str = snn.key_map[k1]
                k2_str = snn.key_map[k2]
                k1_r = any(ele in k1_str for ele in ignore)
                k2_r = any(ele in k2_str for ele in ignore)
                # If neither are Input or Output
                if not k1_r and not k2_r:
                    key_pairs.append((k1, k2, coords_dict[k1_str], coords_dict[k2_str]))
                    global_weight_pairs.append((k1, k2))
                elif k1_r and not k2_r:
                    key_pairs.append((k1, k2, coords_dict[k1_str], coords_dict[k2_str]))
                    global_weight_pairs.append((k1, k2))
                elif k2_r and not k1_r:
                    key_pairs.append((k1, k2, coords_dict[k1_str], coords_dict[k2_str]))
                    global_weight_pairs.append((k1, k2))
                snn.weight_matrix[k1, k1] = 0.0
                snn.weight_matrix[k2, k2] = 0.0
    print("Done!")
    
    global_weight_pairs = tuple(global_weight_pairs)
    key_pairs = tuple(key_pairs)
    global mp_chunk
    mp_chunk = np.rint(np.sqrt(len(global_weight_pairs)))
    # Pre-pickle global_weight_pairs

    global_weight_pairs = pickle.dumps(global_weight_pairs)
    global_weight_pairs = pickle.loads(global_weight_pairs)
    print("Calculating distances as weight:")
    with multiprocessing.Pool(processes=8) as pool:
        for dist, k1, k2 in tqdm(pool.map(EuclidianDistance, key_pairs)):
            snn.weight_matrix[k2, k1] = dist
    print("Done!")
    snn.weight_matrix = (snn.weight_matrix-np.min(snn.weight_matrix))/(np.max(snn.weight_matrix)-np.min(snn.weight_matrix))
    snn.weight_log.append(snn.weight_matrix)
    
    
    snn.RunVision(1)
    outputs = []
    for k in snn.neuron_keys:
        if "Output " in k:
            print(snn.LIFNeurons[k].V)
            outputs.append(np.sum(snn.LIFNeurons[k].V))
    outputs = np.asarray(outputs)
    print(outputs.shape)
    print(snn.n_neurons)
    sh = np.sqrt(snn.n_outputs)
    output_shape = (int(sh), int(sh), 1)
    from sklearn.preprocessing import minmax_scale
    outputs = np.reshape(minmax_scale(outputs, feature_range=(0.0, 255.0)), newshape=output_shape)

    np.save("output.npy", outputs)
    
    # Dump cols and rows
    with open("./logs/ids.txt", "w") as outfile:
        outfile.write(",".join(snn.neuron_keys))
    with open("./logs/debug.log", "w") as outfile:
        for debug_msg in snn.step_debug_log:
            if isinstance(debug_msg, list):
                for process_msg in debug_msg:
                    outfile.writelines(process_msg)
            else:
                outfile.writelines(debug_msg)
    snn.SaveWeightTables()
    #snn.SaveNeuronPotentials()
    #snn.SaveNeuronSpikes()