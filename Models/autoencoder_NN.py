import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

"""trial for autoencoder in one vs all classification"""

"""
this is an attempt to use autoencoders for one vs all classification, following the following example:
https://www.tensorflow.org/tutorials/generative/autoencoder#third_example_anomaly_detection
In the example, the autoencoder is used to recreate ECG data to then detect anomalies. In this trial, the differences
are that:
- we have four classes to detect, not two (anomaly vs not anomaly)
- the ECG data in the example is still a time series. I have not dealt with time series data, at least not until we have
a dataset containing the single heartbeats with uniform length. On the unprocessed dataset, the problem is the nan 
padding

My idea was to use the example as a one vs all classifier, in the case of the code below, it should have been able to 
correctly classify heartbeats of the class '0', but the results are really poor. I played a bit with the layers 
parameters but I believe there is still more to be understood
"""

class AutoEncoderClassification(
    Model
):
    def __init__(self):
        super(AutoEncoderClassification, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(26, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

def separate_dataset(X_train, y_train, class_):
    whole_dataset = pd.concat((X_train, y_train), axis=1)
    mask = np.where(y_train == class_)[0]
    inverse_mask = np.where(y_train != class_)[0]
    normal_dataset = whole_dataset.iloc[mask]
    anomalous_dataset = whole_dataset.iloc[inverse_mask]
    return normal_dataset, anomalous_dataset


if __name__ == "__main__":
    leo_training_data_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data_merged_intervals.csv"
    )
    leo_y_train_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/y_train.csv"
    )
    leo_X_test_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/X_test_data_merged_intervals.csv"
    )

    X_train = pd.read_csv(
        leo_training_data_path
    )
    X_train.drop("Unnamed: 0", axis=1, inplace=True)
    y = pd.read_csv(
        leo_y_train_path
    )
    y.drop("id", axis=1, inplace=True)
    X_test = pd.read_csv(
        leo_X_test_path
    )
    X_test.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(pd.concat((X_train, X_test), axis=0))
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(np.concatenate((X_train, X_test), axis=0))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # don't know if this is necessary
    # X_train = tf.cast(X_train, tf.float32)
    # X_test = tf.cast(X_test, tf.float32)

    autoencoder = AutoEncoderClassification()
    autoencoder.compile(optimizer='Adam', loss='mae')

    normal_data, anomalous_data = separate_dataset(X_train, y, class_=0)
    x_normal = normal_data.iloc[:, 0:-1]
    y_normal = normal_data.iloc[:, -1]

    x_train, x_val, y_train, y_val = train_test_split(x_normal,
                                                      y_normal,
                                                      train_size=0.8,
                                                      random_state=21,
                                                      stratify=y_normal)

    history = autoencoder.fit(x_normal, x_normal,
                              epochs=200,  # change
                              batch_size=512,
                              validation_data=(x_val, x_val),
                              shuffle=True)

    reconstructions = autoencoder.predict(x_normal)
    train_loss = tf.keras.losses.mae(reconstructions, x_normal)

    threshold = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold)

    preds = predict(autoencoder, x_val, threshold)
    print_stats(preds, y_val)
