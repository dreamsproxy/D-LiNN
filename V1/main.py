import numpy as np
from time import time

x = np.random.normal(size=[1, 5])
print(x)

start2 = time()
y1 = ((x > 0) * x)
y2 = ((x <= 0) * x * 0.01)
leaky_way2 = y1 + y2
end2 = time()
print(start2-end2)