import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import keyboard

class AdaptiveLIF:
    def __init__(self, tau_m, R, V_th, V_reset, delta_t, tau_leak):
        self.tau_m = tau_m
        self.R = R
        self.V_th = V_th
        self.V_reset = V_reset
        self.delta_t = delta_t
        self.tau_leak = tau_leak
        self.input_current_value = 0  # Initial input current value

        # Debug trace vars
        self.dV_calculated = False
        self.V_apenned = False
        self.time_apended = False
        self.input_trace = []
        return

    def set_input_current(self, value):
        print(value)
        self.input_current_value = value
        return self.input_current_value
    
    def input_current(self, t):
        return self.input_current_value

    def dV_step(self, input_current, V):
        dV = (-(V[-1] - input_current * self.R) / self.tau_m) * self.delta_t - (V[-1] / self.tau_leak) * self.delta_t
        self.dV_calculated = True

        return dV

    def simulate(self, input_current, sim_ms = 0):

        t = 0
        try:
            #   Initialize the values
            if t == 0:
                time = [0.0]
                V = [self.V_reset]  # Initialize with reset potential
                spikes = [0]
                t += 1
            while True:
                self.dV_calculated = False
                self.time_apended = False

                self.input_trace.append(input_current)
                dV = self.dV_step(input_current, V)

                V.append(V[-1] + dV)
                self.V_apenned = True

                # Check for spike
                if V[-1] >= self.V_th:
                    V[-1] = self.V_reset
                    spikes.append(1)
                else:
                    spikes.append(0)

                time.append(t)
                self.time_apended = True
                t += self.delta_t
                if sim_ms != 0:
                    if t == sim_ms:
                        break
            #sleep(self.delta_t / 1000)  # Sleep to control the simulation speed

        except KeyboardInterrupt:
            # Ensure that time and V are the same length,
            #   Making sure that either V or time updates had executed at the same time constraint
            
            if len(time) < len(V):
                time.append(t)
                self.time_apended = True
                t += self.delta_t
            
            if len(V) < len(time):
                if self.dV_calculated:
                    V.append(V[-1] + dV)
                    self.V_apenned = True
            
            if len(spikes) < len(V):
                # Check for spike
                if V[-1] >= self.V_th:
                    V[-1] = self.V_reset
                    spikes.append(1)
                else:
                    spikes.append(0)

        #print(self.input_trace)
        return time, V, spikes

def plot_neuron(time, V, spikes):
    print(f"total simulated time: {len(time)} ms")

    # Plotting
    plt.figure()
    #plt.subplot(2, 1, 1)
    plt.plot(time, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')

#    plt.subplot(2, 1, 2)
#    plt.plot(time, spikes)
#    plt.xlabel('Time (ms)')
#    plt.ylabel('Spikes')
#    plt.ylim([-0.2, 1.2])

    plt.tight_layout()
    plt.show()
