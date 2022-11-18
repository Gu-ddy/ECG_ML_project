import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

"""
attempt at using B-splines to smoothen the data.
DOESN'T WORK. There seems to be a limit with the package where if the sample data is above a certain
size, the splines return all nan values. 
"""

if __name__ == "__main__":
    leo_training_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train.csv"
    )
    X_train_raw = pd.read_csv(leo_training_path)
    X_train_values = X_train_raw.drop("id", axis=1)

    x_axis = [int(x[1:]) for x in list(X_train_values.columns.values)]
    samples = random.sample(range(1, 5000), 2)
    for sample in samples:
        sample_entries = 5000  # to limit the sample size. For some reason if it's too big, the spline function
        # returns only nans
        data_row = X_train_values.iloc[sample, :]
        data_row = list(data_row)[:sample_entries]
        x_axis = x_axis[:sample_entries]
        spl_1 = UnivariateSpline(x_axis, data_row, s=sample_entries/5)
        spl_2 = UnivariateSpline(x_axis, data_row, s=sample_entries)

        x_input = np.linspace(1, sample_entries, int(sample_entries/4))  # locations on the x-axis where to evaluate
        # the spline function
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(x_axis, data_row)
        ax1.set_title('unsmoothed data')
        ax2.plot(x_input, spl_1(x_input))
        ax2.set_title('smoothed data 1')
        ax3.plot(x_input, spl_2(x_input))
        ax3.set_title('smoothed data 2')
        plt.show()


