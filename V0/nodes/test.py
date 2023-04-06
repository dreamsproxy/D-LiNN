import brian2 as b2
from matplotlib import pyplot as plt
import random
from neurodynex3.tools import input_factory, plot_tools
from neurodynex3.leaky_integrate_and_fire import LIF
from time import time

def main():
    # Neuron model default values
    V_REST = -70 * b2.mV
    V_RESET = -65 * b2.mV
    FIRING_THRESHOLD = -55 * b2.mV
    MEMBRANE_RESISTANCE = 10. * b2.Mohm
    MEMBRANE_TIME_SCALE = 8. * b2.ms
    ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms
    SIM_DURATION = 50 * b2.ms

    #sinusoidal_current = input_factory.get_sinusoidal_current(500, 1500, unit_time=0.1 * b2.ms, amplitude=2.5 * b2.namp, frequency=150 * b2.Hz, direct_current=2. * b2.namp)

    step_current = input_factory.get_step_current(
        t_start=10, t_end=20,
        unit_time=b2.ms,
        amplitude=1.5 * b2.namp
    )

    start = time()
    print(f"started: {start}")
    # run the LIF model
    (state_monitor, spike_monitor) = LIF.simulate_LIF_neuron(
        input_current=step_current,
        simulation_time=SIM_DURATION
    )
    t = (time() - start)
    print(f"Done!\t(took {t})\n")
    print(state_monitor[0])
    print()
    print(spike_monitor.all_values)

    """
    # plot the membrane voltage
    plot_tools.plot_voltage_and_current_traces(
        state_monitor,
        step_current,
        title="Step current",
        firing_threshold=FIRING_THRESHOLD
    )
    print("nr of spikes: {}".format(len(spike_monitor.t)))
    plt.show()
    """
if __name__=="__main__":
    main()