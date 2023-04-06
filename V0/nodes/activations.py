import brian2 as b2
from time import time
import threading
#from matplotlib import pyplot as plt
#import numpy as np
from neurodynex3.leaky_integrate_and_fire import LIF
from neurodynex3.tools import input_factory, plot_tools

spike_array = [0 for x in range(0, 99)]
spike_pos = []
i = 1

#while i <= 3:
#    pos = np.random.randint(0, 99)
#    if not pos in spike_pos:
#        spike_pos.append(pos)
#        i += 1


#print(spike_array)
#print(spike_pos)
#for x in spike_pos:
for x in list([25, 50]):
    spike_array[x] = 1
print(spike_array)
print(spike_pos)
input_current = input_factory.get_spikes_current(
    spike_array,
    1*b2.ms,
    40*b2.uamp,
    append_zero=True
)

# run the LIF model.
# Note: As we do not specify any model parameters, the simulation runs with the default values
(state_monitor,spike_monitor) = LIF.simulate_LIF_neuron(
    input_current=input_current,
    simulation_time = 100 * b2.ms,
    v_rest=-70. * b2.mV,
    v_reset=-65. * b2.mV,
    firing_threshold=-50. * b2.mV,
    membrane_resistance=10. * b2.Mohm,
    membrane_time_scale=8. * b2.ms,
    abs_refractory_period=2. * b2.ms
)

# plot I and vm
plot_tools.plot_voltage_and_current_traces(
    state_monitor,
    input_current,
    title="min input",
    firing_threshold=LIF.FIRING_THRESHOLD
)
print("nr of spikes: {}".format(spike_monitor.count[0]))  # should be 0


"""
start = time()
t = (time() - start)
print(f"Done!\t(took {t})")
!!!!!!!!!...OLD CODE...!!!!!!!!
SNN LIF
While true
CycleID = 0
Check for data
If new data is true
Do n iterations for total pregen time.
If Vm(t) > Vthresh:
Vm(t+1) = Vreset
Else:
Vm(t+1) = Vm(t)+dt*(-Vm(t)-Vrest) + Vinput * Resistance)
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

cycle_id = 0
#   miliseconds
cycle_time = 1000
sim_T = np.arange(1, int(cycle_time), 1)
Vreset = -80.0
Vresting = -75.0
Vm = -75.0
Vthresh = -55.0
tau = 10.0
resistance = 10.3
Vinput = 0.0

VmLog = []

while True:
    for dt in sim_T:
        if Vm > Vthresh:
            Vm = Vreset
        else:
            if dt >= 400 and dt <= 500:
                Vinput = 10.0
            else:
                Vinput = 0.0
            Vm = Vm + dt * ((-(Vm) - Vresting) + Vinput * resistance)
        VmLog.append(Vm)

    cycle_id += 1
    if cycle_id > 12:
        with open("./nodeV.log", "a") as logfile:
            logfile.writelines(VmLog)
    break

print(VmLog)
plt.scatter(sim_T, VmLog)
plt.show()
"""