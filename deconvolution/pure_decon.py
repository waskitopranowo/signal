import numpy as np


def pure_decon(trace, dt, wavelet, prewhitening):
    n = len(trace)
    t = np.arange(0, len(trace), 1) * dt
    wi = np.eye(n, n)

    for i in np.arange(0, wi.shape[1], 1):
        wi[:, i] = np.convolve(wi[:, i], wavelet, 'same')

    prewm = np.eye(n, n) * np.max(wavelet) * prewhitening / 100

    winv = np.linalg.inv(wi + prewm)
    sdecon = np.matmul(winv, trace)

    return t, sdecon
