import numpy as np
import matplotlib.pyplot as plt
from rickerwav import ricker

S = np.loadtxt('data_wno.txt')  # seismic trace
dt = 2  # sampling rate in millisecond

t = np.arange(0, len(S), 1)*dt

wavelet = np.loadtxt('ricker_wavelet.txt')  # (predicted) wavelet
Wi = np.eye(len(S), len(S))

for i in np.arange(0, Wi.shape[1], 1):
    Wi[:, i] = np.convolve(Wi[:, i], wavelet, 'same')

prew = 10  # percentage of pre-whitening
prewm = np.eye(len(S), len(S))*np.max(wavelet)*prew/100

Winv = np.linalg.inv(Wi + prewm)
Sdecon = np.matmul(Winv, S)

plt.subplot(2, 1, 1)
plt.plot(t, S)
plt.title('Before Deconvolution')
plt.subplot(2, 1, 2)
plt.plot(t, Sdecon)
plt.title('After Deconvolution')
plt.show()