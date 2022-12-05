import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def ar_model(signal, lag):
    model = AutoReg(signal, lag)
    model_fit = model.fit()
    return model_fit.params


def arima_model(signal):
    model = ARIMA(signal)
    for p in [0,1,2]:
        for d in [0,1]:
            for q in [0,1,2]:
               model_fit = model.fit()
    print(model_fit.summary)
    return model_fit.model_orders

def cross_intervals(old_data, columns):
    return old_data.loc[:, columns].fillna(0)

if __name__ == "__main__":
    train = False
    if train:
        old_data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data_update.csv"
        data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train.csv"
        current_x_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_Davide.csv"
        new_data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_train_Davide_new.csv"
    else:
        data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test.csv"
        old_data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test_features.csv"
        current_x_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test_Davide.csv"
        new_data_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test_Davide_new.csv"

    # labels_path = "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/y_train.csv"
    data = pd.read_csv(data_path, index_col="id")
    # labels = pd.read_csv(labels_path, index_col="id").to_numpy()

    features = list()
    nr_samples = len(data)
    for patient in range(nr_samples):
        patient_signal = data.iloc[patient].dropna().to_numpy(dtype="float32")
        features.append(list(ar_model(patient_signal, lag=3)))

    features = pd.DataFrame(features)

    old_data = pd.read_csv(old_data_path)
    columns = ["PR_interval", "QT_interval", "QRS_interval", "P_nans", "T_nans", "Q_nans"]
    cross_data = cross_intervals(old_data, columns)
    features = pd.concat((features, cross_data), axis=1)

    Davide_data = pd.read_csv(current_x_path)
    new_features = pd.concat((Davide_data, features), axis=1)
    new_features.to_csv(new_data_path)
    print("yo")
