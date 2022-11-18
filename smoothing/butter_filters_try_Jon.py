import random

import pandas as pd
import numpy as np
#from scipy import signal   #only needed if filtering whole matrix of data (see comment below)
import matplotlib.pyplot as plt
import heartpy as hp

"""
Parameters to play around with:
    cutoff_low (proposed to be between 0.5 and 0.6)
    powerline (either 50 or 60)
    order
"""


def filter_signal(
    data, cutoff_low=0.6, cutoff_high=100, powerline=50, sampling_frequency=300, order=2
):
    """
    Filters low frequencies to get rid of baseline wander, high frequency to get rid of EMG noise and the powerline frequency.

    :param data: dataset we want to filter
    :param cutoff_low: all lower frequencies will be cutoff. get rid of baseline wander (literature: between 0.5Hz and 0.6Hz)
    :param cutoff_high: all higher frequencies will be cutoff. get rid of EMG noise (literature: above 100Hz)
    :param powerline: this frequency will be removed. get rid of powerline frequency (literature: either 50 or 60 Hz, depending on country)
    :param sampling_frequency: the sampling frequency of the digital system
    :param order: order of filter (decides the slope of the frequency cutoff)
    :return: the filtered data
    """
    data = data.to_numpy()

    nr_samples, max_length_signal = data.shape

    low_high_and_powerline_pass_filtered_data = []

    for iteration in range(nr_samples):
        ecg_signal = data[iteration]
        ecg_signal = ecg_signal[~np.isnan(ecg_signal)] #ignore all nans (filling up the row)
        high_pass_filtered_data = hp.filtering.filter_signal(ecg_signal, cutoff=cutoff_low, sample_rate=sampling_frequency, order=order, filtertype='highpass')
        low_and_high_pass_filtered_data = hp.filtering.filter_signal(high_pass_filtered_data, cutoff=cutoff_high, sample_rate=sampling_frequency, order=order, filtertype='lowpass')
        low_high_and_powerline_pass_filtered_data_row = hp.filtering.filter_signal(low_and_high_pass_filtered_data, cutoff=powerline, sample_rate=sampling_frequency, filtertype='notch')
        low_high_and_powerline_pass_filtered_data.append(low_high_and_powerline_pass_filtered_data_row)
        
        
    low_high_and_powerline_pass_filtered_data = pd.DataFrame(low_high_and_powerline_pass_filtered_data)
    
    return (low_high_and_powerline_pass_filtered_data)


if __name__ == "__main__":
    leo_training_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train.csv"
    )
    jon_training_path = ("C:/Users/jonny/Documents/Studium/ETH/Advanced Machine Learning/Projects/Project2/X_train.csv")
    # guglielmo_training_path = ...

    leo_filtered_data_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_filtered.csv"
    )
    jon_filtered_data_path = ("C:/Users/jonny/Documents/Studium/ETH/Advanced Machine Learning/Projects/Project2/X_train_filtered.csv")
    # guglielmo_filtered_data_path = ...

    X_train_raw = pd.read_csv(jon_training_path)
    X_train_values = X_train_raw.drop("id", axis=1)

    X_train_filtered = filter_signal(X_train_values)

    # choose whether to plot the results for some random samples
    plot_results = True
    if plot_results:
        samples = random.sample(range(1, 5000), 2)
        for sample in samples:
            plot_data_raw = X_train_values.iloc[sample, :]
            plot_data_filtered = X_train_filtered.iloc[sample, :]

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(plot_data_raw)
            ax1.set_title('unfiltered data')
            ax2.plot(plot_data_filtered)
            ax2.set_title('filtered data')
            plt.show()

    X_train_filtered.to_csv(jon_filtered_data_path)
    print("er fijo de zaccagni é de zaniolo")



################## Variant that can do it directly with the matrix but doesn't seem to work well because it has to work with nans
    """
    high_pass_filter = signal.butter(
        order,
        cutoff_low,
        'hp',
        fs=sampling_frequency,
        output="sos",
    )
    high_pass_filtered_data = signal.sosfilt(high_pass_filter, data, axis=0)

    low_pass_filter = signal.butter(
        order,
        cutoff_high,
        'lp',
        fs=sampling_frequency,
        output="sos",
    )
    low_and_high_pass_filtered_data = signal.sosfilt(low_pass_filter, high_pass_filtered_data, axis=0)
    
    powerline_filter = signal.butter(
        order,
        (powerline-0.05, powerline-0.05),
        'bs',
        fs=sampling_frequency,
        output="sos",
    )
    low_high_and_powerline_pass_filtered_data = signal.sosfilt(powerline_filter, low_and_high_pass_filtered_data, axis=0)
    
    return pd.DataFrame(low_high_and_powerline_pass_filtered_data)
    """