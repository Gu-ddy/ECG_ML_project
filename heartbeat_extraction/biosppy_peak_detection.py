import numpy as np
from biosppy.signals import ecg
import pandas as pd
import pywt
import neurokit2 as nk
import heartpy as hp
from scipy import stats
from scipy.signal import resample


def wavelet_coeff(signal, family="db1", level=3):
    """
    Computes the first {level} coefficients in wavelet basis expansion to fit the signal
    :param signal: our dataset (filtered)
    :param family: determines what family of wavelet functions we use for the basis expansion
    :param level: number of coefficients for the expansion
    :return: an approximated coefficients array
    """
    family_wavelets = pywt.Wavelet(family)  # perform wavelet transform
    coefficients = pywt.wavedec(data=signal, wavelet=family_wavelets, level=level)
    # coefficients = list of coefficients: [cA_n, cD_n, cD_n-1, …, cD2, cD1]
    # cA_n is approximated coefficients array, the other are detailed.
    # n = level => we have cA_3, cD_3, cD_2, cD_1 for level = 3
    return coefficients[0]


def compute_hos_descriptor(beat, n_intervals, lag):
    """
    dont know what this does but might be helpful
    :param beat:
    :param n_intervals:
    :param lag:
    :return:
    """
    hos_b = np.zeros(((n_intervals - 1) * 2))
    for i in range(0, n_intervals - 1):
        pose = lag * (i + 1)
        interval = beat[int(pose - (lag / 2)) : int(pose + (lag / 2))]

        # Skewness
        hos_b[i] = stats.skew(interval, 0, True)

        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0

        # Kurtosis
        hos_b[(n_intervals - 1) + i] = stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos_b[(n_intervals - 1) + i]):
            hos_b[(n_intervals - 1) + i] = 0.0
    return hos_b


def compute_PQRST(ecg_signal, SAMPLING_RATE=300):
    """
    Detects PQRST peaks in a raw signal using neurokit2. We don't care about R peaks since we already have that info

    Parameters
    ----------
    ecg_signal: np.ndarray
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
    PQRST_peaks = list()
    peak_locations = {}

    peaks, _ = nk.ecg_process(ecg_signal, SAMPLING_RATE)
    peaks_we_seek = [
        "P_Peaks",
        "P_Onsets",
        "P_Offsets",
        "Q_Peaks",
        "R_Onsets",
        "R_Offsets",
        "S_Peaks",
        "T_Peaks",
        "T_Onsets",
        "T_Offsets",
    ]
    for peak_type in peaks_we_seek:
        single_series = peaks.loc[:, f"ECG_{peak_type}"]
        peak_locations[peak_type] = single_series.index[single_series == 1].tolist()

        peak_values = ecg_signal[peak_locations[peak_type]]
        PQRST_peaks += [np.median(peak_values)]
        PQRST_peaks += [stats.median_abs_deviation(peak_values)]

    # cross-peak interval calculation barely works
    """# compute cross-peaktype intervals (PR_interval, PR_segment, QRS, QT, ST)
    cross_intervals = {
        "PR_interval": ["R_Onsets", "P_onsets"],
        "PR_segment": ["R_Onsets", "P_Offsets"],
        "QRS": ["R_Offsets", "R_Onsets"],
        "QT": ["T_Offsets", "R_Onsets"],
        "ST": ["T_Onsets", "R_Offsets"],
    }
    for interval in cross_intervals.values():
        try:
            interval_ = peak_locations[interval[0]] - peak_locations[interval[1]]
            PQRST_peaks += [np.median(interval_)]
            PQRST_peaks += [stats.median_abs_deviation(interval_)]
        except:
            PQRST_peaks += [0]
            PQRST_peaks += [0]
    """
    return PQRST_peaks


def heartpy_features(signal, low_pass=0.6, high_pass=100):
    """
    performs another filter + feature extraction, using the heartpy library
    :param signal: the signal
    :return: features
    """
    feat = list()
    # filter using bandpass butterworth lowpass and highpass filter
    filtered = hp.filter_signal(
        signal, [low_pass, high_pass], sample_rate=300, filtertype="bandpass"
    )
    # resample the signal to len(filtered) * 4 samples
    resampled_signal = resample(filtered, len(filtered) * 4)
    # scale the data to values between 0 and 1024, and process the heart rate data. returns
    # wd: a dict used to store temporary values (we dont need it)
    # m: a dictionary used to store computed measures
    wd, m = hp.process(
        hp.scale_data(resampled_signal), sample_rate=300 * 4, bpmmin=0, bpmmax=200
    )
    # the measures are: bpm, ibi, sdnn, sdsd, rmssd, pnn20, pnn50, hr_mad, sd1, sd2, s, sd1/sd2, breathingrate
    for measure in m.keys():
        if measure not in [
            "sd1/sd2",
            "bpm",
            "hr_mad",
        ]:  # don't withhold measures that are redundant or induce multicolinearity with what we already have
            feat.append(m[measure])
    return feat


def power(signal):  # The value of power is highly influenced by normalization
    return np.dot(signal, signal) / signal.shape[0]


def get_features(signal):
    X = list()
    not_nk_filtered = False
    not_hp_filtered = False
    # ecg.ecg(signal, sampling_rate) extracts relevant signal features from the signal. Returned values that interest us:
    # filtered: the filtered ECG signal
    # rpeaks: the locations of the R peaks
    # templates: extracted heartbeat templates
    # heart_rate: bpm
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(
        signal, 300, show=False
    )
    # ecg.correct_rpeaks() refines the R peak locations to an interval within the tol parameter
    """rpeaks = ecg.correct_rpeaks(
        signal=signal, rpeaks=rpeaks, sampling_rate=300, tol=0.1
    )"""  # was giving errors for data type

    peaks = signal[rpeaks]
    if len(heart_rate) < 2:
        heart_rate = [0, 1]

    # for each patient, we take the following peak statistics
    X.append(np.median(peaks))
    X.append(stats.median_abs_deviation(peaks))
    # other features that might be relevant
    X.append(np.min(peaks))
    X.append(np.max(peaks))
    # compute skewness of the peaks
    X.append(stats.skew(peaks, 0, True))

    # for each patient, we take the following statistics for peak-peak intervals (for some reason not similar to heart_rate
    X.append(np.median(np.diff(rpeaks)))
    X.append(stats.median_abs_deviation(np.diff(rpeaks)))
    X.append(np.min(np.diff(rpeaks)))
    X.append(np.max(np.diff(rpeaks)))
    # X.append(stats.skew(np.diff(rpeaks)[0], 0, True)) # throws a lot of RuntimeWarning: Precision loss occurred in
    # moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical.
    # Results may be unreliable

    X.append(np.median(heart_rate))
    X.append(stats.median_abs_deviation(heart_rate))
    X.append(np.min(heart_rate))
    X.append(np.max(heart_rate))
    X.append(stats.skew(heart_rate, 0, True))

    # stats to capture amount of noise (?) in heartbeat
    X.append(np.sum(filtered - signal))
    X.append(np.median(filtered - signal))
    X.append(stats.median_abs_deviation(filtered - signal))

    # statistics on skewness and deviation of templates row-wise (templates is a matrix)
    X.append(np.median(stats.skew(templates, 0, True)))
    X.append(stats.median_abs_deviation(stats.skew(templates, 0, True)))
    X.append(np.median(stats.median_abs_deviation(templates, axis=0)))
    X.append(stats.median_abs_deviation(stats.median_abs_deviation(templates, axis=0)))

    try:
        X += list(heartpy_features(signal))  # throws a lot of errors
    except:
        X += [0] * 10
        print(f"heartpy filter not working for patient {patient}")
        not_hp_filtered = True
    X += list(
        wavelet_coeff(np.mean(templates, axis=0), family="db1")
    )  # the family is chosen by trial and error

    # no clue what this does
    X += list(
        compute_hos_descriptor(
            np.mean(templates, axis=0), 6, int(len(templates[0]) / 6)
        )
    )

    # get other peak types and intervals with neurokit2.ecg_process
    try:
        X += list(compute_PQRST(signal))
    except:
        X += [0] * 20
        print(f"nk2 filter not working for patient {patient}")
        not_nk_filtered = True

    X.append(power(np.mean(templates, axis=0)))

    X = np.array(X, dtype="float32")
    X[np.isnan(X)] = 0
    return X, not_hp_filtered, not_nk_filtered


if __name__ == "__main__":
    # data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train.csv"
    data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test.csv"
    #labels_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/y_train.csv"
    data = pd.read_csv(data_path, index_col="id")
    #labels = pd.read_csv(labels_path, index_col="id").to_numpy()

    total_not_hp_filtered = list()
    total_not_nk_filtered = list()
    features = list()
    nr_samples = len(data)
    for patient in range(len(data)):
        patient_signal = data.iloc[patient].dropna().to_numpy(dtype="float32")
        features_, not_hp_filtered, not_nk_filtered = get_features(patient_signal)
        features.append(features_)
        if (patient % 100) == 0:
            print(patient, "/", nr_samples, "iterations done.")

        if not_hp_filtered:
            total_not_hp_filtered += [patient]
        if not_nk_filtered:
            total_not_nk_filtered += [patient]

    print(f"total_not_hp_filtered: {total_not_hp_filtered}")
    print(f"total_not_nk_filtered: {total_not_nk_filtered}")
    print(f"len(total_not_hp_filtered): {len(total_not_hp_filtered)}")
    print(f"len(total_not_nk_filtered): {len(total_not_nk_filtered)}")
    X = np.array(features, dtype="object")

    np.save("X_test.npy", X)
