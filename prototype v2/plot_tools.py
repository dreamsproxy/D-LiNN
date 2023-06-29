from matplotlib import pyplot as plt

def plot_neuron(time, V, spikes):
    print(f"total simulated time: {len(time)} ms")

    # Plotting
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')

    plt.subplot(2, 1, 2)
    plt.plot(time, spikes)
    plt.xlabel('Time (ms)')
    plt.ylabel('Spikes')
    plt.ylim([-0.2, 1.2])

    plt.tight_layout()
    plt.show()