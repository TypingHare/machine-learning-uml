# Homework 3 - main file
# COMP.4220 Machine Learning

import itertools, functools
import numpy as np
import matplotlib.pyplot as plt
from regression import LinearRegression, RidgeRegression
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as skStandardScalar


def train_test_split(X, t, test_size=0.2, random_state=None):
    """Splits data into training and testing sets using only NumPy."""

    if random_state:
        np.random.seed(random_state)

    # ---- Part (d) ---- #
    # - Your code here - #

    # 1. Shuffle the data
    indices = np.arange(1)
    X = X
    t = t

    # 2. Split the data
    # split_index = 1
    X_train = X
    X_test = []
    t_train = t
    t_test = []

    return X_train, X_test, t_train, t_test


def standardscalar(x: np.ndarray):
    # ---- Part (b) ---- #
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda _x, y: _x * y, items))
        return np.asarray(features).transpose()


def main():
    # ---- Part (a) ---- #
    # - Your code here - #
    housing = fetch_california_housing()
    X = housing.data
    t = housing.target
    # print(housing.DESCR)

    # ---- Part (b) ---- #
    # - Your code here - #
    Xs = standardscalar(X)
    print(Xs[0:5])
    exit(0)

    # ---- Part (c) ---- #
    # - Your code here - #
    Xss = X
    # print((Xs - Xss))

    # ---- Part (d) ---- #
    # - Your code here - #
    X_train, X_test, t_train, t_test = [], [], [], []

    # ---- Part (k) ---- #
    # - Your code here - #
    # fix the inconsistency between models

    # ---- Part (g, h) ---- #
    # - Your code here - #
    # Model building
    lr = LinearRegression()
    y_lr = []
    print("Linear Regression results")
    print(f"RMSE: {np.inf}")
    print(f"R2: {np.inf}")

    rr = RidgeRegression(lambd=1.0)
    y_rr = []
    print("Ridge Regression results")
    print(f"RMSE: {np.inf}")
    print(f"R2: {np.inf}")

    # ---- Part (i) ---- #
    # - Your code here - #
    lr_sk = skLinearRegression()
    y_lr_sk = []
    print("Sklearn Linear Regression results")
    print(f"RMSE: {np.inf}")
    print(f"R2: {np.inf}")

    rr_sk = skRidge(alpha=1.0)
    y_rr_sk = []
    print("Sklearn Ridge Regression results")
    print(f"RMSE: {np.inf}")
    print(f"R2: {np.inf}")

    # ---- Part (j) ---- #
    # - Your code here - #
    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    # use scatter and plot to show the results
    plt.xlabel("add a proper label")
    plt.ylabel("add a proper label")
    plt.title("add a proper title")

    plt.subplot(2, 2, 2)
    # use scatter and plot to show the results
    plt.xlabel("add a proper label")
    plt.ylabel("add a proper label")
    plt.title("add a proper title")

    plt.subplot(2, 2, 3)
    # use scatter and plot to show the results
    plt.xlabel("add a proper label")
    plt.ylabel("add a proper label")
    plt.title("add a proper title")

    plt.subplot(2, 2, 4)
    # use scatter and plot to show the results
    plt.xlabel("add a proper label")
    plt.ylabel("add a proper label")
    plt.title("add a proper title")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
