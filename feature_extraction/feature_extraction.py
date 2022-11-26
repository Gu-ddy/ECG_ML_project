import numpy as np
import pandas as pd
from scipy import stats

"""
extracts all features from tutorial slides. For each sample (patient), the features are extracted, and their med_abs_dev is 
calculated. 
For a given patient, the function extracts the peak locations from the previously peaks detected peaks, and, for each
peak type, it fetches the amplitudes from the filtered_signal df, and calculates their intervals (e.g. the time 
difference between one R peak and the next one). The function also calculates time differences across peaks, 
specifically the QRS, QT and PR intervals.
    For each peak type and its measurement (i.e. amplitude or interval), a dataframe is created containing each 
    patient's values for said measure. Finally, the median and standard of each patient's values is calculated for 
    each of these dataframes, and the result is put into the final features dataframe.
    
There are a few inconsistencies which may be due to the following facts:
    - for some patients, in the PQRST_peaks dictionary, there were some type of peaks that had a list of locations which 
    was larger length than other types of peaks. E.g. for patient number 17, the 44 P_peaks and P_Onsets were found, but
    45 Q peaks were found. This is a problem when calculating the PR interval, which was here defined by 
    [Q_locations - P_locations].
    Purely from observations, I noticed that most times, when this happened, it was because a P_peak or a P_onset was
    not detected at the beginning, so I fixed this by appending a 0 at the beginning of the P_peaks list.
    This phenomenon is relatively rare, happening roughly once every 200 patients.
    
    - For the nans within a given patient's peak dictionary, e.g. the P_peak dictionary for patient 2, it makes sense to
    remove these values when we're evaluating amplitudes. However, for time differences, it may be more wise to replace
    these nans by another value, for example the mean/median location (?) of the P_peaks for said patients.
    While this isn't a big deal for calculating intervals of the same type of peak (e.g. PP_interval), 
    it causes problems for cross-peak intervals. So far, all the nans have been replaced with 0,
    but I think it makes more sense to replace these nans with the mean/median locations, at least for interval 
    calculations.
"""


def feature_extraction(filtered_signal, peaks):
    """
    Extracts features from the signal (manually).

    Parameters
    ----------
    filtered_signal: pd.DataFrame
        of size (n_samples, max_length_signal)
        if a sample doesn't contain max_length_signal points. It is expected to be filled up with NaNs
    peaks: dictionary
        containing first key with the index and inside dictionaries containing timestamps of PQRST peaks

    Returns
    -------
    features: pd.DataFrame
        of size (n_samples, nr_of_features)
    """

    nr_samples, max_length_signal = filtered_signal.shape

    # the maximum length of the dictionaries in PQRST_peaks is 135. We initialize all the empty dataframes
    max_peak_list_length = 135
    amplitudes_of_P_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    PP_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_R_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    RR_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_S_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    SS_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_Q_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    QQ_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_T_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    TT_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_P_On_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    POPO_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    amplitudes_of_T_Off_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    TOTO_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )

    QRS_interval_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    QT_interval_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )
    PR_intervals_df = pd.DataFrame(
        index=range(nr_samples), columns=list(np.arange(0, max_peak_list_length)), dtype=float
    )

    P_nans_df = pd.DataFrame(
        index=range(nr_samples), columns=["nans"], dtype=float
    )
    T_nans_df = pd.DataFrame(
        index=range(nr_samples), columns=["nans"], dtype=float
    )
    S_nans_df = pd.DataFrame(
        index=range(nr_samples), columns=["nans"], dtype=float
    )
    Q_nans_df = pd.DataFrame(
        index=range(nr_samples), columns=["nans"], dtype=float
    )

    amplitudes_of_P = []
    PP_intervals = []
    amplitudes_of_R = []
    RR_intervals = []
    amplitudes_of_S = []
    SS_intervals = []
    amplitudes_of_Q = []
    QQ_intervals = []
    amplitudes_of_T = []
    TT_intervals = []
    amplitudes_of_P_On = []
    POPO_intervals = []
    amplitudes_of_T_Off = []
    TOTO_intervals = []


    QRS_intervals = []
    QT_intervals = []
    PR_intervals = []

    # todo: merge redundant predictions (XX_intervals) using a statistic or a regression
    # create a feature map that allows us to copy the least amount of code possible and use for loops
    feature_map = {
        "P": {
            "amplitudes": {"list": amplitudes_of_P, "dataframe": amplitudes_of_P_df},
            "intervals": {"list": PP_intervals, "dataframe": PP_intervals_df},
            "nans": {"list": [], "dataframe": P_nans_df},
        },
        "R": {
            "amplitudes": {"list": amplitudes_of_R, "dataframe": amplitudes_of_R_df},
            "intervals": {"list": RR_intervals, "dataframe": RR_intervals_df},
        },
        "S": {
            "amplitudes": {"list": amplitudes_of_S, "dataframe": amplitudes_of_S_df},
            "intervals": {"list": SS_intervals, "dataframe": SS_intervals_df},
            "nans": {"list": [], "dataframe": S_nans_df},
        },
        "Q": {
            "amplitudes": {"list": amplitudes_of_Q, "dataframe": amplitudes_of_Q_df},
            "intervals": {"list": QQ_intervals, "dataframe": QQ_intervals_df},
            "nans": {"list": [], "dataframe": Q_nans_df},
        },
        "T": {
            "amplitudes": {"list": amplitudes_of_T, "dataframe": amplitudes_of_T_df},
            "intervals": {"list": TT_intervals, "dataframe": TT_intervals_df},
            "nans": {"list": [], "dataframe": T_nans_df},
        },
        "P_Onsets": {
            "amplitudes": {"list": amplitudes_of_P_On, "dataframe": amplitudes_of_P_On_df},
            "intervals": {"list": POPO_intervals, "dataframe": POPO_intervals_df},
        },
        "T_Offsets": {
            "amplitudes": {"list": amplitudes_of_T_Off, "dataframe": amplitudes_of_T_Off_df},
            "intervals": {"list": TOTO_intervals, "dataframe": TOTO_intervals_df}
        },
    }

    # create a dict where the keys are the feature names to go into the final df, and the value the table from which
    # that feature must be extracted
    tables_dict = {
        "P_amp": amplitudes_of_P_df,
        "P_amp_med_abs_dev": amplitudes_of_P_df,
        "P_nans": P_nans_df,
        "PP_interval": PP_intervals_df,
        "PP_interval_med_abs_dev": PP_intervals_df,
        "R_amp": amplitudes_of_R_df,
        "R_amp_med_abs_dev": amplitudes_of_R_df,
        "RR_interval": RR_intervals_df,
        "RR_interval_med_abs_dev": RR_intervals_df,
        "S_amp": amplitudes_of_S_df,
        "S_amp_med_abs_dev": amplitudes_of_S_df,
        "S_nans": S_nans_df,
        "SS_interval": SS_intervals_df,
        "SS_interval_med_abs_dev": SS_intervals_df,
        "Q_amp": amplitudes_of_Q_df,
        "Q_amp_med_abs_dev": amplitudes_of_Q_df,
        "Q_nans": Q_nans_df,
        "QQ_interval": QQ_intervals_df,
        "QQ_interval_med_abs_dev": QQ_intervals_df,
        "T_amp": amplitudes_of_T_df,
        "T_amp_med_abs_dev": amplitudes_of_T_df,
        "T_nans": T_nans_df,
        "TT_interval": TT_intervals_df,
        "TT_interval_med_abs_dev": TT_intervals_df,
        "P_On_amp": amplitudes_of_P_On_df,
        "P_On_amp_med_abs_dev": amplitudes_of_P_On_df,
        "POPO_interval": POPO_intervals_df,
        "POPO_interval_med_abs_dev": POPO_intervals_df,
        "T_Off_amp": amplitudes_of_T_Off_df,
        "T_Off_amp_med_abs_dev": amplitudes_of_T_Off_df,
        "TOTO_interval": TOTO_intervals_df,
        "TOTO_interval_med_abs_dev": TOTO_intervals_df,
        "QRS_interval": QRS_interval_df,
        "QRS_interval_med_abs_dev": QRS_interval_df,
        "QT_interval": QT_interval_df,
        "QT_interval_med_abs_dev": QT_interval_df,
        "PR_interval": PR_intervals_df,
        "PR_interval_med_abs_dev": PR_intervals_df,
    }

    # access dictionary row for row
    for it in range(nr_samples):
        for peak_type, features in feature_map.items():
            # get and modify peak locations
            peak_type_key = f"ECG_{peak_type}"
            if peak_type not in ["P_Onsets", "T_Offsets"]:
                peak_type_key += "_Peaks"
            idx = np.array(peaks[it][peak_type_key])

            single_df = filtered_signal.iloc[it, :]

            # todo: check why amplitudes throw nans at certain peak locations
            # amplitude of peaks
            features["amplitudes"]["list"] = single_df[idx[~np.isnan(idx)].astype(int)]

            # time between peaks
            nan_difference = np.diff(idx)
            nan_dropped_difference = nan_difference[~np.isnan(nan_difference)]
            features["intervals"]["list"] = nan_dropped_difference.astype(int)

            if "nans" in features.keys():
                features["nans"]["list"] = [sum([1 for x in np.isnan(idx) if x])]

            for measure in features.values():
                # update the dataframe
                measure["dataframe"].iloc[it, range(len(measure["list"]))] = measure[
                    "list"
                ]

        # calculate other intervals. Since we're dealing with differences between different types of peak locations,
        # the nans are filled to a defined variable num_to_fill. Further incentivises the use of median as statistic,
        # as we're bound to have large outliers in the differences. It affects med_abs_dev a lot tho

        Q_indices = np.array(peaks[it]["ECG_Q_Peaks"])
        """
        num_to_fill = np.nanmean(Q_indices)
        Q_indices = np.nan_to_num(Q_indices, nan=num_to_fill)
        Q_indices = Q_indices.astype(int)"""

        S_indices = np.array(peaks[it]["ECG_S_Peaks"])
        T_offsets_indices = np.array(peaks[it]["ECG_T_Offsets"])
        P_onset_indices = np.array(peaks[it]["ECG_P_Onsets"])

        try:
            QRS_intervals = (S_indices - Q_indices)[~np.isnan(S_indices - Q_indices)]
            QT_intervals = (T_offsets_indices - Q_indices)[~np.isnan(T_offsets_indices - Q_indices)]
            PR_intervals = (Q_indices - P_onset_indices)[~np.isnan(Q_indices - P_onset_indices)]
        # catch error in case there is a mismatch in array size between two of the arrays we use to define intervals.
        # The new values are appended because I was working with long lists before, but this is no longer necessary.
        # However, it still works
        except ValueError:
            if len(Q_indices) < len(S_indices):
                Q_indices = np.concatenate(([0], Q_indices))
                print(f"caught iteration {it}")
            elif len(Q_indices) < len(T_offsets_indices):
                Q_indices = np.concatenate(([0], Q_indices))
                print(f"caught iteration {it}")
            elif len(Q_indices) < len(P_onset_indices):
                Q_indices = np.concatenate(([0], Q_indices))
                print(f"caught iteration {it}")
            elif len(P_onset_indices) < len(Q_indices):
                P_onset_indices = np.concatenate(([0], P_onset_indices))
                print(f"caught iteration {it}")
            else:
                print(f"iteration {it}, still not caught it")

            QRS_intervals = (S_indices - Q_indices)[~np.isnan(S_indices - Q_indices)]
            QT_intervals = (T_offsets_indices - Q_indices)[~np.isnan(T_offsets_indices - Q_indices)]
            PR_intervals = (Q_indices - P_onset_indices)[~np.isnan(Q_indices - P_onset_indices)]

        QRS_interval_df.iloc[it, range(len(QRS_intervals))] = QRS_intervals.astype(int)
        QT_interval_df.iloc[it, range(len(QT_intervals))] = QT_intervals.astype(int)
        PR_intervals_df.iloc[it, range(len(PR_intervals))] = PR_intervals.astype(int)

    features = pd.DataFrame(index=range(nr_samples), columns=list(tables_dict.keys()))

    for key, table in tables_dict.items():
        if "med_abs_dev" in key:
            features[key] = stats.median_abs_deviation(table, axis=1, nan_policy='omit')
        else:
            features[key] = table.median(axis=1)

    return features


if __name__ == "__main__":
    leo_filtered_data_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_filtered.csv"
    )
    jon_filtered_data_path = ...
    guglielmo_filtered_data_path = ...

    leo_peaks_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/PQRST_peaks.npy"
    )
    jon_peaks_path = ...
    guglielmo_peaks_path = ...

    leo_features_df_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data_update.csv"
    )
    jon_features_df_path = ...
    guglielmo_features_df_path = ...

    X_train_filtered = pd.read_csv(leo_filtered_data_path)
    peaks = np.load(leo_peaks_path, allow_pickle="TRUE").item()

    features = feature_extraction(X_train_filtered, peaks)
    features.to_csv(leo_features_df_path)
