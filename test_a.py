import numpy as np
from matplotlib import pyplot as plt


t = np.linspace(0, 1.0, 200)
xlow = np.sin(2 * np.pi * 5 * t)
plt.plot(t,xlow)
plt.show()
