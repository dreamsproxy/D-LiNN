import numpy as np

steps = 10  # Number of steps
initial_value = 100
decay_rate = 0.9

# Calculate the values
values = []
for step in range(steps):
    value = initial_value * decay_rate ** step
    values.append(value)

# Print the values
for i, value in enumerate(values):
    print("Step {}: {:.2f}".format(i, value))