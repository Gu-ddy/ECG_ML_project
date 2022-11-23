import numpy as np
import pandas as pd

def feature_extraction(filtered_signal, peaks, statistic = "median"):
    """"
    Extracts features from the signal (manually).

    Parameters
    ----------
    filtered_signal: np.ndarray 
        of size (n_samples, max_length_signal)
        if a sample doesn't contain max_length_signal points. It is expected to be filled up with NaNs
    peaks: dictionary
        containing first key with the index and inside dictionaries containing timestamps of PQRST peaks
    statistic: string
        the statistic to use to transform, for a given patient, all of his peaks of a given type into a single value

    Returns
    -------
    features: np.ndarray
        of size (n_samples, nr_of_features)
    """

    nr_samples, max_length_signal = filtered_signal.shape

    # the maximum length of the dictionaries in PQRST_peaks is 135
    amplitudes_of_R_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    RR_intervals_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    amplitudes_of_S_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    SS_intervals_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    amplitudes_of_Q_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    QQ_intervals_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    amplitudes_of_T_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    TT_intervals_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    QRS_interval_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    QT_interval_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))
    PR_intervals_df = pd.DataFrame(index=range(nr_samples), columns=list(np.arange(0, 135)))

    amplitudes_of_R = []
    RR_intervals = []
    amplitudes_of_S = []
    SS_intervals = []
    amplitudes_of_Q = []
    QQ_intervals = []
    amplitudes_of_T = []
    TT_intervals = []

    QRS_intervals = []
    QT_intervals = []
    PR_intervals = []

    feature_map = {"R": {"amplitudes": {"list": amplitudes_of_R, "dataframe": amplitudes_of_R_df},
                         "intervals": {"list": RR_intervals, "dataframes": RR_intervals_df},
                         },
                   "S": {"amplitudes": {"list": amplitudes_of_S, "dataframe": amplitudes_of_S_df},
                         "intervals": {"list": SS_intervals, "dataframe": SS_intervals_df}
                         },
                   "Q": {"amplitudes": {"list": amplitudes_of_Q, "dataframe": amplitudes_of_Q_df},
                         "intervals": {"list": QQ_intervals, "dataframe": QQ_intervals_df}
                         },
                   "T": {"amplitudes": {"list": amplitudes_of_T, "dataframe": amplitudes_of_T_df},
                         "intervals": {"list": TT_intervals, "dataframe": TT_intervals_df}
                         },
                   }

    tables_dict = {"R_amp": amplitudes_of_R_df,
                   "R_amp_std": amplitudes_of_R_df,
                   "RR_interval": RR_intervals_df,
                   "RR_interval_std": RR_intervals_df,
                   "S_amp": amplitudes_of_S_df,
                   "S_amp_std": amplitudes_of_S_df,
                   "SS_interval": SS_intervals_df,
                   "SS_interval_std": SS_intervals_df,
                   "Q_amp": amplitudes_of_Q_df,
                   "Q_amp_std": amplitudes_of_Q_df,
                   "QQ_interval": QQ_intervals_df,
                   "QQ_interval_std": QQ_intervals_df,
                   "T_amp": amplitudes_of_T_df,
                   "T_amp_std": amplitudes_of_T_df,
                   "TT_interval": TT_intervals_df,
                   "TT_interval_std": TT_intervals_df,
                   "QRS_interval": QRS_interval_df,
                   "QRS_interval_std": QRS_interval_df,
                   "QT_interval": QT_interval_df,
                   "QT_interval_std": QT_interval_df,
                   "PR_interval": PR_intervals_df,
                   "PR_interval_std": PR_intervals_df,
                   }


    for it in range(nr_samples):  # access dictionary row for row
        for peak_type, features in feature_map.items():
            # amplitude of peaks
            peak_type_key = f"ECG_{peak_type}_Peaks"
            idx = np.array(peaks[it][peak_type_key])
            # todo: check if we should just ignore or fill nans. Might not be great for interval calculation
            idx = idx[~np.isnan(idx)] #ignore all nans
            idx = idx.astype(int)
            amplitudes = features["amplitudes"]["list"]
            single_df = filtered_signal.iloc[it, :]
            amplitudes.append(single_df[idx])

            #time between peaks
            intervals = features["intervals"]["list"]
            intervals.append(np.diff(idx))

        # calculate other intervals. Since we're dealing with differences between different types of peak locations,
        # the nans are filled to a defined variable num_to_fill. Further incentivises the use of median as statistic,
        # as we're bound to have large outliers in the differences
        num_to_fill = 0

        Q_indices = np.array(peaks[it]["ECG_Q_Peaks"])
        Q_indices = np.nan_to_num(Q_indices, nan=num_to_fill)
        Q_indices = Q_indices.astype(int)

        S_indices = np.array(peaks[it]["ECG_S_Peaks"])
        S_indices = np.nan_to_num(S_indices, nan=num_to_fill)
        S_indices = S_indices.astype(int)

        T_offsets_indices = np.array(peaks[it]["ECG_T_Offsets"])
        T_offsets_indices = np.nan_to_num(T_offsets_indices, nan=num_to_fill)
        T_offsets_indices = T_offsets_indices.astype(int)

        P_onset_indices = np.array(peaks[it]["ECG_P_Onsets"])
        P_onset_indices = np.nan_to_num(P_onset_indices, nan=num_to_fill)
        P_onset_indices = P_onset_indices.astype(int)

        QRS_intervals.append(S_indices - Q_indices)
        QT_intervals.append(T_offsets_indices - Q_indices)
        PR_intervals.append(Q_indices - P_onset_indices)

        for features in feature_map.values():
            for measure in features.values():
                measure["dataframe"][it, range(len(measure["list"]))] = measure["list"]

        QRS_interval_df[it, range(len(QRS_intervals))] = QRS_intervals
        QT_interval_df[it, range(len(QT_intervals))] = QT_intervals
        PR_intervals_df[it, range(len(PR_intervals))] = PR_intervals

    features = pd.DataFrame(index=range(nr_samples), columns=list(tables_dict.keys()))
    for key, table in tables_dict.items():
        if "std" in key:
            features[key] = table.std(axis=1)
        else:
            features[key] = table.median(axis=1)
    
    return features

if __name__ == "__main__":
    leo_filtered_data_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_filtered.csv"
    )
    leo_peaks_path = (
        '/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/PQRST_peaks.npy'
    )
    leo_features_df_path = (
        '/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data.csv'
    )

    X_train_filtered = pd.read_csv(leo_filtered_data_path)
    peaks = np.load(leo_peaks_path,allow_pickle='TRUE').item()

    features = feature_extraction(X_train_filtered, peaks)
    features.to_csv(leo_features_df_path)