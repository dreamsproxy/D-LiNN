import numpy as np
from matplotlib import pyplot as plt

def default_pars(**kwargs):
    pars = {}

    # typical neuron parameters#
    pars['V_th'] = -55.     # spike threshold [mV]
    pars['V_reset'] = -75.  # reset potential [mV]
    pars['tau_m'] = 10.     # membrane time constant [ms]
    pars['g_L'] = 10.       # leak conductance [nS]
    pars['V_init'] = -75.   # initial potential [mV]
    pars['E_L'] = -75.      # leak reversal potential [mV]
    pars['tref'] = 2.       # refractory time (ms)

    # simulation parameters #
    pars['T'] = 400.  # Total duration of simulation [ms]
    pars['dt'] = .1   # Simulation time step [ms]

    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    # Vector of discretized time points [ms]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

    return pars

def run_LIF(pars, Iinj, stop=False):
    """
    Simulate the LIF dynamics with external input current

    Args:
    pars       : parameter dictionary
    Iinj       : input current [pA]. The injected current here can be a value
                    or an array
    stop       : boolean. If True, use a current pulse

    Returns:
    rec_v      : membrane potential
    rec_sp     : spike times
    """

    # Set parameters
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    V_init, E_L = pars['V_init'], pars['E_L']
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size
    tref = pars['tref']

    # Initialize voltage
    v = np.zeros(Lt)
    v[0] = V_init

    # Set current time course
    Iinj = Iinj * np.ones(Lt)

    # If current pulse, set beginning and end to 0
    if stop:
        Iinj[:int(len(Iinj) / 2) - 1000] = 0
        Iinj[int(len(Iinj) / 2) + 1000:] = 0

    # Loop over time
    rec_spikes = []  # record spike times
    tr = 0.  # the count for refractory duration

    for it in range(Lt - 1):
        if tr > 0:  # check if in refractory period
            v[it] = V_reset  # set voltage to reset
            tr = tr - 1 # reduce running counter of refractory period

        elif v[it] >= V_th:  # if voltage over threshold
            rec_spikes.append(it)  # record spike event
            v[it] = V_reset  # reset voltage
            tr = tref / dt  # set refractory time

        # Calculate the increment of the membrane potential
        dv = (-(v[it] - E_L) + Iinj[it] / g_L) * (dt / tau_m)

        # Update the membrane potential
        v[it + 1] = v[it] + dv

    # Get spike times in ms
    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes


def plot_volt_trace(pars, v, sp):
    """
    Plot trajetory of membrane potential for a single neuron

    Expects:
    pars   : parameter dictionary
    v      : volt trajetory
    sp     : spike train

    Returns:
    figure of the membrane potential trajetory for a single neuron
    """

    V_th = pars['V_th']
    dt, range_t = pars['dt'], pars['range_t']
    if sp.size:
        sp_num = (sp / dt).astype(int) - 1
        v[sp_num] += 20  # draw nicer spikes
    fig, axes = plt.subplots(1, 1)
    axes.plot(pars['range_t'], v, 'b')
    axes.axhline(V_th, 0, 1, color='k', ls='--')
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('V (mV)')
    axes.legend(['Membrane\npotential', r'Threshold V$_{\mathrm{th}}$'],
                loc=[1.05, 0.75])
    axes.set_ylim([-80, -40])
    axes.set_xlim([0, 400])
    axes.xaxis.set_ticks(range(0, 400, 10))
    axes.axes
    axes.grid(visible = True)
    plt.show()

# Get parameters
pars = default_pars(T=500)

# Simulate LIF model
v, sp = run_LIF(pars, Iinj=100, stop=True)

# Visualize
plot_volt_trace(pars, v, sp)
