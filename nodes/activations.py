"""
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
"""
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
