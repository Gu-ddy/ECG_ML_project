import random

import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


def filter_signal(
    data, butterworth_frequency=10, critical_frequency=5000, sampling_frequency=1000
):
    """
    :param data: dataset we want to filter
    :param butterworth_frequency: the order of the butterworth filter we create
    :param critical_frequency: the critical frequency. For values that are too large, it returns an overflow error
    :param sampling_frequency: the sampling frequency of the digital system
    :return: the filtered data
    """
    filter = signal.butter(
        N=butterworth_frequency,
        Wn=0.99,  # critical frequency ## if parameter fs is specified, then it can be a number (lower than fs/2)
        btype="lp",
        # fs=sampling_frequency,  ## can be defined if we want to set a sampling frequency
        output="sos",
    )
    filtered_data = signal.sosfilt(filter, data, axis=0)
    return pd.DataFrame(filtered_data)


if __name__ == "__main__":
    leo_training_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train.csv"
    )
    # jon_training_path = ...
    # guglielmo_training_path = ...

    leo_filtered_data_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_filtered.csv"
    )
    # jon_filtered_data_path = ...
    # guglielmo_filtered_data_path = ...

    X_train_raw = pd.read_csv(leo_training_path)
    X_train_values = X_train_raw.drop("id", axis=1)

    X_train_filtered = filter_signal(X_train_values)

    # choose whether to plot the results for some random samples
    plot_results = True
    if plot_results:
        samples = random.sample(range(1, 5000), 5)
        for sample in samples:
            plot_data_raw = X_train_values.iloc[sample, :]
            plot_data_filtered = X_train_filtered.iloc[sample, :]

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(plot_data_raw)
            ax1.set_title('unfiltered data')
            ax2.plot(plot_data_filtered)
            ax2.set_title('filtered data')
            plt.show()

    X_train_filtered.to_csv(leo_filtered_data_path)
    print("er fijo de zaccagni é de zaniolo")
