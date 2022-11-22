import numpy as np
import pandas as pd

def feature_extraction(filtered_signal, peaks):
    """"
    Extracts features from the signal (manually).

    Parameters
    ----------
    filtered_signal: np.ndarray 
        of size (n_samples, max_length_signal)
        if a sample doesn't contain max_length_signal points. It is expected to be filled up with NaNs
    peaks: dictionary
        containing first key with the index and inside dictionaries containing timestamps of PQRST peaks

    Returns
    -------
    features: np.ndarray
        of size (n_samples, nr_of_features)
    """

    nr_samples, max_length_signal = filtered_signal.shape

    amplitudes_of_R = []
    times_between_R_peaks = []
    amplitudes_of_S = []
    times_between_S_peaks = []
    for it in range(nr_samples): #access dictionary row for row
        #amplitude of R peak
        idx = np.array(peaks[it]["ECG_R_Peaks"])
        idx = idx[~np.isnan(idx)] #ignore all nans
        idx = idx.astype(int)
        amplitudes_of_R.append(filtered_signal[it, idx])

        #time between R peaks
        times_between_R_peaks.append(np.diff(idx))

        #amplitude of S peak
        idx = np.array(peaks[it]["ECG_S_Peaks"])
        idx = idx[~np.isnan(idx)] #ignore all nans
        idx = idx.astype(int)
        amplitudes_of_S.append(filtered_signal[it, idx])
        
        #time between S peaks
        times_between_S_peaks.append(np.diff(idx))

    amplitudes_of_R = pd.DataFrame(amplitudes_of_R) #to account for different row lengths first convert to df
    amplitudes_of_R = np.array(amplitudes_of_R) #then to ndarray. rows are filled up with nans

    times_between_R_peaks = pd.DataFrame(times_between_R_peaks)
    times_between_R_peaks = np.array(times_between_R_peaks)

    amplitudes_of_S = pd.DataFrame(amplitudes_of_S)
    amplitudes_of_S = np.array(amplitudes_of_S)
    
    times_between_S_peaks = pd.DataFrame(times_between_S_peaks)
    times_between_S_peaks = np.array(times_between_S_peaks)

    median_amplitude_of_R = np.nanmedian(amplitudes_of_R, axis=1)
    std_amplitude_of_R = np.nanstd(amplitudes_of_R, axis=1)
    median_time_between_R_peaks =np.nanmedian(times_between_R_peaks, axis=1)
    std_time_between_R_peaks = np.nanstd(times_between_R_peaks, axis=1)
    median_amplitude_of_S = np.nanmedian(amplitudes_of_S, axis=1)
    std_amplitude_of_S = np.nanstd(amplitudes_of_S, axis=1)
    median_time_between_S_peaks =np.nanmedian(times_between_S_peaks, axis=1)
    std_time_between_S_peaks = np.nanstd(times_between_S_peaks, axis=1)
    

    features = np.stack((median_time_between_S_peaks, std_time_between_S_peaks, median_amplitude_of_R, std_amplitude_of_R, median_time_between_R_peaks, std_time_between_R_peaks, median_amplitude_of_S, std_amplitude_of_S), axis=-1)
    return features