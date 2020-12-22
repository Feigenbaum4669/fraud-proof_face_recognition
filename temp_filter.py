from tracking import tracking
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz
from matplotlib.mlab import find
from parabolic import parabolic
from sklearn.decomposition import PCA
from thd_calc import THDN

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y



def freq_from_AC(sig, fs):
    """Estimate frequency using autocorrelation
    
    Pros: Best method for finding the true fundamental of any repeating wave, 
    even with strong harmonics or completely missing fundamental
    
    Cons: Not as accurate, currently has trouble with finding the true peak
    
    """
    # Calculate circular autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    corr = signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    
    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    # Also could zero-pad before doing circular autocorrelation.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs/px

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

ps = tracking("video1.mp4");

N=len(ps)
no_of_signals=len(ps[0][1])
raw_signals = [[0 for x in range(no_of_signals)] for y in range(N)] 

for i in range(N):
	#print("probe N: "+repr(i)+" "+repr(N)+"\n")
	for j in range(no_of_signals):
		
		#print("signal: "+repr(j)+"\n")
		raw_signals[i][j]=ps[i][1][j][1]
		#print(repr(raw_signals[i][j])+"\n")


    # Sample rate and desired cutoff frequencies (in Hz).
fs = 60.0
lowcut = 0.75
highcut = 5.0
order = 5
T=N/fs
raw_signals_np = np.array(raw_signals).reshape(N,no_of_signals );
filtered_signals=np.empty((N,no_of_signals))
for i in range(no_of_signals):
	in_signal_i=raw_signals_np[:,i]
	out_signal_i = butter_bandpass_filter(in_signal_i, lowcut, highcut, fs, order)
	for j in range(N):
        	filtered_signals[j][i]=out_signal_i[j]

t = np.linspace(0, T, N, endpoint=False)
a = 2
y = filtered_signals[:,0]
plt.plot(t, y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()



#print(filtered_signals)
pca = PCA()
pca.fit(filtered_signals)
pca_res = pca.transform(filtered_signals)

for i in range(no_of_signals):
	plt.plot(pca_res[:,i])
	plt.show()
	#ps = np.abs(np.fft.fft(pca_res[:,i]))**2
	#time_step = 1/fs

	#freqs = np.fft.fftfreq(N, time_step)
	#idx   = np.argsort(freqs)

	#plt.plot(freqs[idx], ps[idx])
	#plt.show()

	#plt.plot(autocorr(pca_res[:,i]))
	#plt.show()
	print(repr(60*freq_from_AC(pca_res[:,i],fs)))
	THDN(pca_res[:,i],fs)


