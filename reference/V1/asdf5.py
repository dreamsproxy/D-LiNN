import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Requested by Alan on 2023-04-20
# to modify PHD/White paper code to move Neuron
# into it's own class

class Neuron:
    """
    Neuron with typical parameters

    Implement to be as similar to the original as possible.

    Note: the original implementation assumes each time frame are
    of exact same. See method `step` implementation for more detail
    """

    # Tau(m)    membrane time constant [ms]
    # 
    # Explain 1:
    #   time required for membrane potential to hit 0;
    # 
    # Explain 2:
    #   anouunt of time (ms) until membrane potential (mV)
    #   will take to reach resting potential);

    def __init__(self, 
            ### typical neuron parameters ###
            V_th: float     = -55.0, # spike threshold [mV]
            V_reset: float  = -75.0, # reset potential [mV]
            tau_m: float    =  10.0, # membrane time constant [ms]
            R_m: float      =  20.0,  # mebrane resistance [ohm]
            g_L: float      =  10.0, # leak conductance [nS]
            V_init: float   = -65.0, # initial potential [mV]
            V_L: float      = -75.0, # leak reversal potential [mV]
            tref: float     =   2.0, # refractory time (ms)
        ) -> None:

        self.V_th: float = V_th
        self.V_reset: float = V_reset
        self.tau_m: float = tau_m
        self.R_m: float = R_m
        self.g_L: float = g_L
        self.V_init: float = V_init
        self.V_L: float = V_L
        self.tref: float = tref

        self._voltage_right_now: float = V_init
        """Voltage of this neuron, right now. Used when emitting to other neuron(s)"""
        
        self._refractory_steps_left: float = 0.0
        """
        refractory steps until input to this neuron is accepted again.
        """

        return
    
    def step_custom(self, input_voltage: float = 300.0, dt: float = 0.1) -> Tuple[float, bool]:
        """Step over a given input_voltage and timeframe

        Returns: (Neuron Voltage (float), Neuron Spiked (bool))
        
        Note: currently the property `_refractory_steps_left` is used to count down
        number of step before accepting changes.
        """
        should_spike: bool = False

        """Forces to run on 0.1 ms all the time"""
        # do nothing for a given time after exceeding threshold
        if self._refractory_steps_left > 0:
            self._voltage_right_now = self.V_reset
            self._refractory_steps_left = self._refractory_steps_left - 1.0 # deduct 1 wait step
        
        # reset voltage and record spike event
        elif self._voltage_right_now >= self.V_th:
            self._voltage_right_now = self.V_reset
            self._refractory_steps_left = self.tref / dt # Number of steps to wait
            should_spike = True
        
        # calculate the input current with the neuron's resistance
        input_current = (input_voltage * self.R_m)
        
        # calculate the increment of the membrane potential
        dv: float = (-(self._voltage_right_now-self.V_L) + input_current/self.g_L) * (dt/self.tau_m)
        
        # update the membrane potential
        self._voltage_right_now = self._voltage_right_now + dv

        if should_spike:
            return (self._voltage_right_now * self.R_m)
        return (self._voltage_right_now, should_spike)

def run_simulation() -> None:

    # # simulation setup
    # ## typical neuron parameters is used
    neuron: Neuron = Neuron()
    
    # ## simulation parameters
    simulation_duration: float = 500.0    # Total duration of simulation [ms]
    simulation_time_step: float = 0.1     # Simulation time step [ms]
    
    # ## Vector of discretized time points [ms]
    range_t: List[float] = np.arange(0, simulation_duration, simulation_time_step)
    
    # Initialize voltage and current
    voltage_over_time: List[float] =   0.0 * np.ones(range_t.size)
    current_over_time: List[float] = 300.0 * np.ones(range_t.size)

    # record time at which spike occurs
    spiked_at_time_list: List[float] = []

    # simulate the LIF dynamics
    for idx, (time_since_start, in_current) in enumerate(zip(range_t, current_over_time)):
        
        voltage_over_time[idx], spiked = neuron.step_custom(input_voltage=in_current, dt=simulation_time_step)
        if spiked:
            spiked_at_time_list.append(time_since_start)
    
    plt.plot(range_t, voltage_over_time, 'b')
    plt.xlim(0, 100)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')

    plt.show()


if __name__ == "__main__":
    run_simulation()
    """output should be current, but we have Voltage"""