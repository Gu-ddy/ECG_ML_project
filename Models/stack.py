import pandas as pd
import numpy as np

import xgboost
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import kernels

# Euler doesnt have skopt package
"""from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer"""


def naive_bayes_learn(x, y, inner):
    model = GaussianNB()

    select = SelectKBest(f_classif)
    param_grid = {"selector__k": np.arange(50, 86)}

    pipe = Pipeline([("selector", select), ("model", model)])
    tune = GridSearchCV(pipe, param_grid, scoring="f1_micro", cv=inner, verbose=0)

    tune.fit(x, y)
    print(f"naive_bayes score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def random_forest_learn(x, y, inner):
    model = RandomForestClassifier(n_estimators=150)
    param_grid = {
        "class_weight": ["balanced", None],
        "max_features": ["log2", "sqrt", None],
    }

    tune = GridSearchCV(model, param_grid, scoring="f1_micro", cv=inner, verbose=3)

    tune.fit(x, y)
    print(f"random_forest score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def xgboost_learn(x, y, inner):
    model = xgboost.XGBClassifier()
    param_grid = {
        "n_estimators": [2, 3, 4],
        "tree_method": ["auto", "gpu_hist"],
        "sampling_method": ["uniform", "gradient_based"],
    }

    tune = GridSearchCV(model, param_grid, scoring="f1_micro", cv=inner, verbose=0)
    tune.fit(x, y)
    print(f"xgboost score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def svm_learn(x, y, inner):
    model = SVC()

    param_grid = {
        "C": np.arange(1e1, 4e2, 1e1),
        "kernel": ["rbf", "poly"],
        "class_weight": [None, "balanced"],
        "degree": [4, 5, 6],
    }

    # COULD ADD DECISION FUNCTION !
    # ,'gamma': Real (1e-2,1e1, prior='log-uniform')} Don't know why it doesn't work with gamma

    tune = GridSearchCV(model, param_grid, scoring="f1_micro", cv=inner, verbose=0)
    tune.fit(x, y)
    print(f"svm_core: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def gaussian_processes_learn(x, y, inner):
    model = GaussianProcessClassifier()

    param_grid = {
        "kernel": [
            kernels.Matern() + kernels.ConstantKernel(),
            kernels.RationalQuadratic() + kernels.ConstantKernel(),
        ]
    }

    tune = GridSearchCV(model, param_grid, scoring="f1_micro", cv=inner, verbose=0)

    tune.fit(x, y)
    print(f"gaussian_processes score: {tune.best_score_}\nParams: {tune.best_params_}")
    return tune


def stack_learn(x_path, y_path, models):
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path, index_col="id")
    y = np.ravel(y)
    predictions = pd.DataFrame(
        columns=list(models.keys())
    )  # also keep a table with predictions in case we want to add NN predictions

    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    estimators = []

    for model_name, model in models.items():
        tune = model(x, y, inner)
        preds = tune.predict(x)
        predictions.loc[:, model_name] = preds
        estimators.append(tuple((model_name, tune.best_estimator_)))

    stack = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(multi_class="ovr")
    )
    stack.fit(x, y)

    score = np.average(
        cross_val_score(X=x, y=y, estimator=stack, cv=inner, scoring="f1_micro")
    )
    print(f"stack_score: {score}")

    return stack


x_train_path = (
    "/cluster/home/lbarberi/X_train_Davide.csv"
)
y_train_path = "/cluster/home/lbarberi/y_train.csv"

models = {"random_forest": random_forest_learn,
          "naive_bayes": naive_bayes_learn,
          "xgb": xgboost_learn,
          "svm": svm_learn,
          "gaussian_processes": gaussian_processes_learn,
          }
# todo: add KNeighboursClassifier and its function

stack = stack_learn(x_train_path, y_train_path, models)

x_test_path = (
    "/cluster/home/lbarberi/X_test_Davide.csv"
)
x_test = pd.read_csv(x_test_path)
final_predictions = stack.predict(x_test)
submission_dict = {"id": x_test.index, "y": final_predictions}
final_predictions = pd.DataFrame(submission_dict)
final_predictions.to_csv("/cluster/home/lbarberi/final_predictions.csv")

print("o")
