# coded by waskito pranowo, universitas pertamina
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

data = np.loadtxt('data_nonstat1.txt')
dt = 1
time = data[np.arange(0, data.shape[0], dt), 0]
sig = data[np.arange(0, data.shape[0], dt), 1]

windowms = 101  # gabor transform window in samples
overlap = 0.75 # window overlapping (in fraction)

window = int(np.round(windowms/dt))
zpad = int(np.floor(window/2))  # number of zero padding
sigzpad = np.hstack((np.zeros(zpad), sig, np.zeros(zpad)))

noverlap = overlap*window
nshift = int(1 - noverlap)
stft = []
for i in np.arange(0, len(sig), nshift):
    movwind = np.zeros(len(sigzpad))
    movwind[i:i + window] = windows.gaussian(window, window/6)
    sigwin = np.multiply(sigzpad, movwind)
    sigwinf = abs(np.fft.fft(sigwin))
    stft.append(sigwinf)

stft = np.asarray(stft).T
N = np.ceil(stft.shape[0]/2)
stft = stft[0:int(N), :]
f = np.linspace(0, 1, N)*1000/(2*dt)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(time, sig)
plt.xlim(min(time), max(time))
plt.subplot(2, 1, 2)
plt.imshow(stft, extent=[min(time), max(time), max(f), min(f)], aspect='auto', interpolation='bilinear')
plt.show()
