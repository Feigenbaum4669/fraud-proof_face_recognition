import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import freqz


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


if __name__ == "__main__":
    

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 60.0
    lowcut = 0.75
    highcut = 5.0
    order = 5

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    w, h = freqz(b, a, worN=512)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter signal.

    T = 3
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 2
    f0 = 3
    x = 1 * np.sin(2 * np.pi * 0.5 * np.sqrt(t))
    x += 1 * np.cos(2 * np.pi * 50 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + 0.11)
    x += 3 * np.cos(2 * np.pi * 100 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()

