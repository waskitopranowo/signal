import numpy as np
import matplotlib.pyplot as plt
from deconvolution import pure_decon

S = np.loadtxt('data_wno.txt')  # seismic trace
dt = 2  # sampling rate in millisecond

wavelet = np.loadtxt('ricker_wavelet.txt')  # (predicted) wavelet
prewhitening = 10  # pre-whitening percentage

t, Sdecon = pure_decon(S, dt, wavelet, prewhitening)

plt.subplot(2, 1, 1)
plt.plot(t, S)
plt.title('Before Deconvolution')
plt.subplot(2, 1, 2)
plt.plot(t, Sdecon)
plt.title('After Deconvolution')
plt.show()
