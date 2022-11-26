"""
gets as input a df with the features extracted from feature_extraction.py. Returns the same df but where the XX_interval
columns are merged into one/four variables. This is to avoid redundancy in our variables and minimize multicolinearity.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold


def merge_intervals(features_df, peak_types, y):
    """
    :param features_df: the dataframe containing the features
    :param peak_types: a list containing peak_types X for which we have an 'XX_interval' column in the df
    :return: a dataframe where the XX_interval variables are merged into one
    """
    interval_column_names = ["Unnamed: 0"]
    interval_column_names += [x + f"{x}_interval" for x in peak_types]
    interval_column_names += [x + f"{x}_interval_med_abs_dev" for x in peak_types]
    intervals_df = features_df[interval_column_names].dropna()
    y = y.iloc[intervals_df["Unnamed: 0"]]

    # run a random forest regression on intervals_df to predict the classes encoded in the y variable
    model = RandomForestClassifier()
    param_grid = {
        "n_estimators": list(np.arange(80, 120, 10)),
        "min_samples_split": [2, 3]
    }
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    tune = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=inner,
        verbose=-1,
        # error_score="raise",
    )
    tune.fit(intervals_df, y)
    y_pred = tune.predict_proba(intervals_df)

    new_features_df = features_df.iloc[intervals_df["Unnamed: 0"]].drop(interval_column_names, axis=1)
    new_features_df.append(y_pred, axis=1)
    return new_features_df

if __name__ == "__main__":
    leo_features_df_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data_update.csv"
    )
    jon_features_df_path = ...
    guglielmo_features_df_path = ...

    leo_new_df_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/features_data_merged_intervals.csv"
    )
    jon_new_df_path = ...
    guglielmo_new_df_path = ...

    leo_y_train_path = (
        "/Users/leonardobarberi/Desktop/ETH/Semester_1/AML/task2/y_train.csv"
    )

    y = pd.read_csv(leo_y_train_path)
    features_df = pd.read_csv(leo_features_df_path)
    peak_types = ["P", "R", "Q", "T", "S", "PO", "TO"]

    new_features = merge_intervals(features_df, peak_types, y)
    new_features.to_csv(leo_new_df_path)