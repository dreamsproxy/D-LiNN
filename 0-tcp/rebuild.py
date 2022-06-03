from matplotlib import pyplot as plt
from numpy import zeros
from numpy import arange
import numpy as np
from datetime import datetime
import asyncio
from pprint import pprint
import time

# Leaky Integrate and Fire
T = 100                     # ms; prepare the temporal sequence for fake sync
dt = 0.1                    # ms; timestep
spike_threshold = -55       # mV; to fire
spike_delta = 0.5           #   IDK what this is
resting_potential = -75     # mV; in which it waits for signal
neuron_potential = -55      # mV; default when spawned
membrane_resistance = 1     # kOhm; resistance of ion channels
membrane_capacitance = 10   # uF; nano farads
init_time = datetime.utcnow()

#   Membrane time constant
#       It is defined as the amount of TIME it takes for the
#       change in potential to reach 63% of its final value.
#   milliseconds
membrane_tau = membrane_resistance * membrane_capacitance

#   Refractory Period
#   milliseconds
membrane_tau_refractory = 4


input_mV = 1
potential_log = []
time_log = []
pregenerated_time_sequence = arange(0, dt + T, dt)
potential_log = zeros(len(pregenerated_time_sequence))

for i, ms in enumerate(pregenerated_time_sequence):
    #print(i)
    if ms > 0:
        potential_log[i] = potential_log[i-1] + (-potential_log[i-1] + input_mV * membrane_resistance) / membrane_tau * dt
    if ms >= spike_threshold:
        potential_log[i] += spike_delta
        resting_potential = ms + membrane_tau_refractory
    time_log.append(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    time.sleep(float(dt/100))

print(len(time_log))
print(len(potential_log))

neural_data = np.array([time_log, potential_log])
plt.plot(time_log, potential_log)
plt.xticks(np.arange(0.000, 0.100, 0.005))
plt.grid(visible = True)
plt.show()
pprint(neural_data)