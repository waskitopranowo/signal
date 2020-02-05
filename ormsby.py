# scripted by Waskito Pranowo

def ormsbyfilt(lowcut, lowpass, highpass, highcut, dt, L):
    import numpy as np
    L = np.floor(L/2)

    t = np.linspace(-L,L,L*2+1)*dt/1000
    A1 = ((np.pi*highcut)**2)/(np.pi*highcut - np.pi*highpass)*(np.sinc(highcut*t))**2
    A2 = ((np.pi*highpass)**2)/(np.pi*highcut - np.pi*highpass)*(np.sinc(highpass*t))**2
    A3 = ((np.pi*lowpass)**2)/(np.pi*lowpass - np.pi*lowcut)*(np.sinc(lowpass*t))**2
    A4 = ((np.pi*lowcut)**2)/(np.pi*lowpass - np.pi*lowcut)*(np.sinc(lowcut*t))**2

    A = (A1 - A2) - (A3 - A4)
    A = A/np.sum(A)
    return A