import numpy as np
from numpy import linalg


def wiener(input, desired_output, trace, prewhitening):
    A = np.correlate(input, input, 'full')
    A = A[int(np.floor(len(A)/2)):None]
    A[0] = A[0]*(1 + prewhitening)
    X = np.correlate(desired_output, input, 'full')
    X = X[int(np.floor(len(X)/2)):None]
    T = linalg.toeplitz(A)
    Tinv = np.linalg.inv(T)
    Winop = np.matmul(Tinv, X)  # Wiener Operator
    Sdeconw = np.convolve(trace, Winop)  #deconvolved signal by using Wiener
    Sdeconw = Sdeconw[0:len(trace)]
    return Sdeconw

def spiking(input, trace, prewhitening):
    desired_output = np.zeros(len(input))
    desired_output[0] = 1

    A = np.correlate(input, input, 'full')
    A = A[int(np.floor(len(A)/2)):None]
    A[0] = A[0]*(1 + prewhitening)
    X = np.correlate(desired_output, input, 'full')
    X = X[int(np.floor(len(X)/2)):None]
    T = linalg.toeplitz(A)
    Tinv = np.linalg.inv(T)
    Winop = np.matmul(Tinv, X)  # Wiener Operator
    Sdeconw = np.convolve(trace, Winop)  #deconvolved signal by using Wiener
    Sdeconw = Sdeconw[0:len(trace)]
    return Sdeconw
