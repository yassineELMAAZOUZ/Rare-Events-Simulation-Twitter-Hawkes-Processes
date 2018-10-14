import numpy as np
import matplotlib.pyplot as plt
import time as tm
import os



data1 = np.loadtxt("estimations.txt")

plt.boxplot(data1)
plt.show()

