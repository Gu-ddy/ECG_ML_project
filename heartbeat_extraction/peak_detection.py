import numpy as np
import neurokit2 as nk

"""
Parameters to play around with:
    SAMPLING_RATE: might be given, same for all the samples?
    method for ecg_peaks
    method for ecg_delineate
"""

SAMPLING_RATE = 500 #seems to be quite important

def peak_detection(filtered_signal):
    """
    Detects PQRST peaks in a filtered signal.

    Parameters
    ----------
    filtered_signal: np.ndarray 
        of size (n_samples, max_length_signal)
        if a sample doesn't contain max_length_signal points. It is expected to be filled up with NaNs

    Returns
    -------
    PQRST_peaks: Dictionary of Dictionaries
        first dictionary contains key id of sample (integer, row_nr)
            inside there are the keys "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets"
            each then contains the indices at which the specific Peak occurs.
            example: {0: {"ECG_P_Peaks": [345, 678,..], "ECG_Q_Peaks": [45, 376,..], ..}, 1:{..}, ..}
    """
    nr_samples, max_length_signal = filtered_signal.shape

    PQRST_peaks = {} #initialize empty dictionary

    for iteration in range(nr_samples): #neurokit only works with vectors and not with matrices
        ecg_signal = filtered_signal[iteration]
        ecg_signal = ecg_signal[~np.isnan(ecg_signal)] #ignore all nans (filling up the row)
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=SAMPLING_RATE) #different methods are possible
        __, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=SAMPLING_RATE, method="peak") #different methods are possible
        PQRST_peaks[iteration] = waves_peak

    return PQRST_peaks