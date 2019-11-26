import numpy as np
import matplotlib.pyplot as plt
from att_amp import att_amp

S = np.loadtxt('data_teapot_2ms.txt')  # seismic trace
dt = 2  # sampling rate in millisecond

window = 20  # in sample
method = 'absolute_avg'

Sm = att_amp(S, window, method)

t = np.arange(0, Sm.shape[0], 1)*dt
x = np.arange(0, Sm.shape[1], 1)
plt.imshow(Sm, aspect='auto', interpolation='bilinear', extent=[min(x), max(x), max(t), min(t)])
plt.colorbar()
plt.show()
