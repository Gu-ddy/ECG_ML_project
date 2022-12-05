import pandas as pd
import numpy as np

import xgboost
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFpr, SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import kernels
from mpi4py import MPI

# Euler doesnt have skopt package
"""from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer"""

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


def naive_bayes_learn(x, y, inner):
    model = GaussianNB()
    param_grid = {"var_smoothing": [1e-6, 1e-4, 1e-2, 1, 10]}
    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"naive_bayes score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def random_forest_learn(x, y, inner):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "class_weight": ["balanced", None],
        "max_features": ["log2", "sqrt", None],
        "max_samples": [0.8, None],
        "n_estimators": [80, 100, 120, 150]
    }

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"random_forest score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def xgboost_learn(x, y, inner):
    model = xgboost.XGBClassifier()
    param_grid = {
        "max_depth": [6, 8, 10],
        "subsample": [0.8, 1],
        "n_estimators": [80, 100, 120, 150]
    }

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )
    tune.fit(x, y)
    print(f"xgboost score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def svm_learn(x, y, inner):
    model = SVC()

    param_grid = {
        "C": np.arange(1, 102, 20),
        "kernel": ["rbf", "poly"],
        "class_weight": [None, "balanced"],
        "degree": [4, 5, 6],
        "probability": [True],
    }

    # COULD ADD DECISION FUNCTION !
    # ,'gamma': Real (1e-2,1e1, prior='log-uniform')} Don't know why it doesn't work with gamma

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )
    tune.fit(x, y)
    print(f"svm_score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def gaussian_processes_learn(x, y, inner):
    model = GaussianProcessClassifier()

    param_grid = {
        "kernel": [
            kernels.RationalQuadratic() + kernels.ConstantKernel(),
        ]
    }

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"gaussian_processes score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def bagging_classifier_learn(x, y, inner):
    model = BaggingClassifier()

    param_grid = {"max_features": np.arange(20, 81, 10)}

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"bagging_classifier score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def KNeighbours_learn(x, y, inner):
    model = KNeighborsClassifier()

    param_grid = {
        "n_neighbors": np.arange(3, 8, 1),
        "weights": ["uniform", "distance"],
    }

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"K_Neighbours score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def MLP_learn(x, y, inner):
    model = MLPClassifier()

    param_grid = {
        "hidden_layer_sizes": [(60, 20), (50, 30), (180, 60, 20), (180, 20), (180, 90, 30)],
        "max_iter": [600, 800, 1000],
    }

    tune = GridSearchCV(
        model, param_grid, scoring="f1_micro", cv=inner, verbose=0, n_jobs=-1
    )

    tune.fit(x, y)
    print(f"MLP score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def vote_learn(x, y, models):
    y = np.ravel(y)

    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    estimators = []

    for model_name, model_ in models.items():
        model_learner = model_[0]
        model_rank = model_[1]

        if RANK == model_rank:
            tune = model_learner(x, y, inner)
            sendend = {model_name: tune.best_params_}
            comm.isend(sendend, dest=SIZE - 1, tag=model_rank)
            print(f"model_learner {model_name} has been sent")

    if RANK == SIZE - 1:
        for single_rank in range(SIZE - 1):
            request = comm.irecv(source=single_rank, tag=single_rank)
            received = request.wait()
            print(f"estimator params received from rank {single_rank}")
            model_name = list(received.keys())[0]
            params = received[model_name]
            model = models[model_name][2]

            estimator = model(**params)
            estimators.append(tuple((model_name, estimator)))

        print(f"estimators: {estimators}")

        voting_param_grid = {
            "voting": ["hard", "soft"],
        }
        voter = VotingClassifier(estimators=estimators)
        tune = GridSearchCV(
            voter,
            param_grid=voting_param_grid,
            n_jobs=-1,
            verbose=3,
        )
        tune.fit(x, y)

        score = np.average(
            cross_val_score(
                X=x,
                y=y,
                estimator=tune.best_estimator_,
                cv=inner,
                scoring="f1_micro",
                n_jobs=-1,
                verbose=3,
            )
        )
        print(f"stack_score: {score}, best params: {tune.best_params_}")

        return tune.best_estimator_




x_train_path = "/cluster/home/lbarberi/X_train_Davide_new.csv"
x_train = pd.read_csv(x_train_path)
y_train_path = "/cluster/home/lbarberi/y_train.csv"
y_train = pd.read_csv(y_train_path, index_col="id")
x_test_path = "/cluster/home/lbarberi/X_test_Davide_new.csv"
x_test = pd.read_csv(x_test_path)

scaler = StandardScaler()
scaler.fit(pd.concat((x_train, x_test), axis=0))
x_train = pd.DataFrame(scaler.transform(x_train))
x_test = pd.DataFrame(scaler.transform(x_test))

models = {
    "random_forest": [random_forest_learn, 0, RandomForestClassifier],
    "xgb": [xgboost_learn, 1, xgboost.XGBClassifier],
    # "svm": [svm_learn, 2, SVC],
    # "MLP_classifier": [MLP_learn, 3, MLPClassifier],
    # "KNeighboursClassifier": [KNeighbours_learn, 3, KNeighborsClassifier],
    # "naive_bayes": [naive_bayes_learn, 4, GaussianNB],
    # "bagging_classifier": [bagging_classifier_learn, 3, BaggingClassifier],
    # "gaussian_processes": [gaussian_processes_learn, 6, GaussianProcessClassifier],
}

stack = vote_learn(x_train, y_train, models)

if RANK == SIZE - 1:
    final_predictions = stack.predict(x_test)
    submission_dict = {"id": x_test.index, "y": final_predictions}
    final_predictions = pd.DataFrame(submission_dict)
    final_predictions.to_csv("/cluster/home/lbarberi/final_predictions_voting.csv")

    print("o")
