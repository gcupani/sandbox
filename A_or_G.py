import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as sfft
from scipy.io import wavfile

def note(f, r=1e5, d=1.0):
    r = int(r)
    return (np.sin(2*np.pi * np.arange(r*d) * f/r)).astype(np.float32)

def plot(t, n, name, mode='time', labels=None):
    """ Plot a note in the time domain """
    n = np.array(n)
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    if mode == 'time':
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([0,1e-2])
    elif mode == 'spec':
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([0,2000])        
    if len(n.shape) == 1:
        ax.plot(t, n)
    elif len(n.shape) == 2:
        for i, ni in enumerate(n):
            try:
                ax.plot(t, ni, label=labels[i])
            except:
                ax.plot(t, ni)
    if len(n.shape) == 2:
        ax.legend()
    plt.show()
    fig.savefig(name, format='png')
    
# number of sample points
p = 1e5

# sample step
s = 1/p

# rate in Hz
r = int(1/s)

# duration 
d = p*s

# Time range
t = np.linspace(0, p*s, p)

# Frequency range
f = np.linspace(0, 0.5/s, p/2)

# Note frequency
f_A = 440  # A
f_G = 391.995  # G

# Notes in time domain
n_A = np.sin(f_A * t * 2*np.pi)  # A
n_G = np.sin(f_G * t * 2*np.pi)  # G
n_rA = np.sin(f_A * np.mod(t, 1/f_G) * 2*np.pi)  # Repeated A

# Note spectra
sp_A = 2.0/p * np.abs(sfft.fft(n_A)[:int(p)//2])  # A
sp_G = 2.0/p * np.abs(sfft.fft(n_G)[:int(p)//2])  # G
sp_rA = 2.0/p * np.abs(sfft.fft(n_rA)[:int(p)//2])  # Repeated A

# Time domain plots
plot(t, n_A, "A_time.png")
plot(t, n_G, "G_time.png")
plot(t, n_rA, "rA_time.png")
plot(t, [n_A, n_G, n_rA], "all_time.png", labels=['A', 'G', 'A repeated'])

# Spectrum plots
plot(f, sp_A, "A_spec.png", mode='spec')
plot(f, sp_G, "G_spec.png", mode='spec')
plot(f, sp_rA, "rA_spec.png", mode='spec')
plot(f, [sp_A, sp_G, sp_rA], "all_spec.png", mode='spec', 
     labels=['A', 'G', 'A repeated'])
     
# Audio file
wavfile.write('A.wav', r, n_A)
wavfile.write('G.wav', r, n_G)
wavfile.write('rA.wav', r, n_rA)