import numpy as np


def att_amp(traces, window, method):
    if len(traces.shape) == 1:
        traces = np.asarray([traces]).T

    ns = traces.shape[0]
    no = traces.shape[1]

    zp = np.int(np.floor(window / 2))
    s2 = np.vstack((np.zeros((zp, no)), traces, np.zeros((zp, no))))
    sm = np.zeros((ns, no))

    for i in np.arange(0, ns, 1):
        if method == 'average':
            sm[i, :] = np.mean(s2[np.arange(0, window, 1) + i, :], axis=0)
        elif method == 'absolute_avg':
            sm[i, :] = np.mean(np.abs(s2[np.arange(0, window, 1) + i, :]), axis=0)
        elif method == 'RMS':
            sm[i, :] = np.sqrt(np.mean((s2[np.arange(0, window, 1) + i, :]) ** 2, axis=0))

    return sm
