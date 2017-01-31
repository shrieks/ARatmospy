import numpy as np
import pyfftw as pf
#import scipy.fftpack as sf

def gen_avg_per_unb(closedloop_data, interval_length, halfover=False, meanrem=False,
                    hanning=False, hamming=False, nofftw=False):
    """
    this uses a blackman window to generate an unbiased (low leakage) periodogram.

    closedloop_data is a 1D numpy array of phase
    interval_length sets the number of samples per interval (a power of 2 is a good choice)
    halfover flag does half-overlapping
    """
    total_len = closedloop_data.size
    per_len   = interval_length

    ## Check for flags
    # Remove the mean if the 'mean remove' flag is set
    if meanrem:
        mydata = closedloop_data - np.mean(closedloop_data)
    else:
        mydata = closedloop_data

    # check interval length
    if halfover:
        num_intervals = np.floor(total_len/(per_len/2.0)) - 1
        start_indices = np.arange(num_intervals, dtype=float) * per_len / 2
    else:
        num_intervals = np.floor(total_len/per_len) 
        start_indices = np.arange(num_intervals, dtype=float) * per_len

    ind = np.arange(per_len, dtype=float)

    # check window requested
    if hanning:
        window = 0.50 - 0.50 * np.cos(2 * np.pi * ind/(per_len-1))
    elif hamming:
        window = 0.54 - 0.46 * np.cos(2 * np.pi * ind/(per_len-1))
    else:
        window = 0.42 - 0.50 * np.cos(2 * np.pi * ind/(per_len-1)) + \
                        0.08 * np.cos(4 * np.pi * ind/(per_len-1))
    ## Done with flag options

    ## PSD calculation
    psd = np.zeros(per_len)   # this is a float array by default
    for a in np.arange(num_intervals):
        this_start = start_indices[a]
        if nofftw:
            psd = psd + np.abs(np.fft.fft(mydata[this_start:(this_start+per_len)]*window)/per_len)**2
        else:
            psd = psd + np.abs(pf.interfaces.numpy_fft.fft(mydata[this_start:(this_start+per_len)] * 
                                                           window)/per_len)**2
            
    psd = psd / num_intervals
    win2 = window**2
    psd = psd * per_len / win2.sum()

    return psd

