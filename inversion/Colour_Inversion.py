import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

data = np.loadtxt('AI', skiprows=30)
time = data[:, 0]   # time
AI = data[:, 1]     # Acoustic Impedance
trace = np.loadtxt('trace', skiprows=7)
trace = trace[:, 1] # seismic trace
sr = 4              # sampling rate in ms
doff = 10           # delta offset

fmin = 10   # minimum frequency
fmax = 70   # maximum frequency
L = 60     # sample number of operator length
sm = 35    # frequency smoother (Hz)

AIf = np.fft.fft(AI)
AIf = AIf[0:int(np.round(len(AIf)/2))]
freq = np.linspace(0, 1, len(AIf))*1000/(2*sr)

AIflog = np.log10(abs(AIf[1:None]))
freqlog = np.log10(freq[1:None])

m = np.polyfit(freqlog, AIflog, 1)

A = np.zeros((len(freq)))
A[1:None] = freq[1:None]**m[0]*10**m[1]

Sf = np.fft.fft(trace)
Sf = Sf[0:int(np.round(len(Sf)/2))]

df = freq[1] - freq[0]
Nsm = np.int(sm/df)

Sfs = np.convolve(abs(Sf), np.ones(Nsm)/Nsm, 'same')

n1 = np.min(np.where(freq >= fmin))
n2 = np.max(np.where(freq <= fmax))

F = np.zeros((len(A)))
F[n1:n2+1] = A[n1:n2+1]/(Sfs[n1:n2+1] + 10**(-2))
F = np.hstack((F, np.flipud(F[1:None])))

Ft = np.fft.ifft(F).real
Ft = np.hstack((Ft[int(np.round(len(Ft)/2)):None], Ft[0:int(np.round(len(Ft)/2))]))

n = len(Ft) - L
Ft = Ft[int(n/2):len(Ft)-int(n/2)]
Ft = signal.hilbert(Ft).imag

traces = np.loadtxt('volve_arb_PGC.txt')
nt = traces.shape[0]
no = traces.shape[1]

RelI = np.zeros((nt, no))   # relative Impedance
for i in np.arange(0, no, 1):
    RelI[:, i] = np.convolve(traces[:, i], Ft, 'same')

time = np.arange(0, nt, 1)*sr
off = np.arange(0, no, 1)*doff
plt.imshow(RelI, aspect='auto', cmap='seismic',
           extent=[np.min(off), np.max(off), np.max(time), np.min(time)],
           vmin=np.min(RelI)*0.5, vmax=np.max(RelI)*0.9)
plt.colorbar()
plt.show()
