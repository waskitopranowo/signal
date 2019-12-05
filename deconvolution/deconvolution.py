import numpy as np


def pure_decon(trace, dt, wavelet, prewhitening):
    #  trace is trace input
    #  dt is sampling rate in ms
   
    n = len(trace)
    t = np.arange(0, len(trace), 1) * dt
    wi = np.eye(n, n)

    for i in np.arange(0, wi.shape[1], 1):
        wi[:, i] = np.convolve(wi[:, i], wavelet, 'same')

    prewm = np.eye(n, n) * np.max(wavelet) * prewhitening / 100

    winv = np.linalg.inv(wi + prewm)
    sdecon = np.matmul(winv, trace)

    return t, sdecon

def spec_div_decon(trace, dt, fdes, smoothwindow, prew):
    #  trace is trace input
    #  dt is sampling rate in ms
    #  fdes is desired bandpass frequency
    #  smoothwindow is width of moving average for spectum smooting. It is used for predicting wavelet's spectrum
    #  prew is prewhitening

    Sf = np.fft.fft(trace)
    Sfhalf = Sf[0:int(np.ceil(len(Sf) / 2))]
    f = np.linspace(0, 1000 / (2 * dt), len(Sfhalf))  # frequency in Hz

    f1 = fdes[0]
    f2 = fdes[1]
    f3 = fdes[2]
    f4 = fdes[3]

    if1 = np.where(abs(f - f1) == np.min(abs(f - f1)))[0]
    if2 = np.where(abs(f - f2) == np.min(abs(f - f2)))[0]
    if3 = np.where(abs(f - f3) == np.min(abs(f - f3)))[0]
    if4 = np.where(abs(f - f4) == np.min(abs(f - f4)))[0]

    P1 = np.zeros((if1))
    P2 = np.linspace(0, 1, if2 - if1)
    P3 = np.ones(if3 - if2)
    P4 = np.linspace(1, 0, if4 - if3)
    P5 = np.zeros((len(f) - if4))

    Des = np.hstack((P1, P2, P3, P4, P5))  # desired spectrum

    window = np.ones(smoothwindow) / smoothwindow
    Wf = np.convolve(abs(Sfhalf), window, 'same')  # spectrum of wavelet

    Ophalf = Des / (Wf + np.mean(Wf) * prew)
    Op = np.hstack((Ophalf, np.flipud(Ophalf[1:None])))

    Sdeconf = np.multiply(Op, Sf)
    Sdecon = np.fft.ifft(Sdeconf).real

    return Sdecon, f, Ophalf
