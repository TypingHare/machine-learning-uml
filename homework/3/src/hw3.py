# Homework 3 - main file
# COMP.4220 Machine Learning

import itertools, functools

import numpy
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

    # 1. Shuffle the data
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_train = int(len(indices) * (1 - test_size))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # 2. Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    t_train = t[train_indices]
    t_test = t[test_indices]

    return X_train, X_test, t_train, t_test


def standardscalar(x: np.ndarray):
    """Standardizes features by removing the mean and scaling to unit variance
    (z-score normalization).
    """
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


def rmse(target: np.ndarray, prediction: np.ndarray):
    return np.sqrt(np.mean((target - prediction) ** 2))


def r2():
    pass


def main():
    # ---- Part (a) ---- #
    housing = fetch_california_housing()
    X = housing.data
    t = housing.target
    # print(housing.DESCR)

    # ---- Part (b) ---- #
    Xs = standardscalar(X)

    # ---- Part (c) ---- #
    Xss = skStandardScalar().fit_transform(X)
    # Check if `Xs` and `Xss` are identical
    print(numpy.array_equal(Xs, Xss))  # Should be `True`

    # ---- Part (d) ---- #
    X_train, X_test, t_train, t_test = train_test_split(X, t)

    # ---- Part (k) ---- #
    # Fix the inconsistency between models
    # "When features have very different scales, features with larger scales can
    # dominate the model training. Standardization makes all features have
    # mean=0 and std=1, so they contribute more equally to the model"
    scalar = skStandardScalar()
    scalar.fit(X_train)  # Calculate the mean and deviation
    X_train = scalar.transform(X_train)  # Transform using the formula:
    X_test = scalar.transform(X_test)  # X' = (X - µ) / σ
    t_mean = np.mean(t_test)  # Center the targets
    t_train = t_train - t_mean
    t_test = t_test - t_mean
    # "When features are standardized (mean=0, std=1) but target isn't, the
    # scale mismatch can cause numerical instability in matrix operations
    # (like pseudoinverse)"

    # ---- Part (g, h) ---- #
    # Model building
    lr = LinearRegression()
    lr.fit(X_train, t_train)
    y_lr = lr.predict(X_test)
    print("[Linear Regression Results]")
    print(f"RMSE: {root_mean_squared_error(t_test, y_lr)}")
    print(f"R2: {r2_score(t_test, y_lr)}")
    print()

    rr = RidgeRegression(_lambda=1.0)
    rr.fit(X_train, t_train)
    y_rr = rr.predict(X_test)
    print("[Ridge Regression Results]")
    print(f"RMSE: {root_mean_squared_error(t_test, y_rr)}")
    print(f"R2: {r2_score(t_test, y_rr)}")
    print()

    # ---- Part (i) ---- #
    lr_sk = skLinearRegression()
    lr_sk.fit(X_train, t_train)
    y_lr_sk = lr_sk.predict(X_test)
    print("[Sklearn Linear Regression Results]")
    print(f"RMSE: {root_mean_squared_error(t_test, y_lr_sk)}")
    print(f"R2: {r2_score(t_test, y_lr_sk)}")
    print()

    rr_sk = skRidge(alpha=1.0)
    rr_sk.fit(X_train, t_train)
    y_rr_sk = rr_sk.predict(X_test)
    print("[Sklearn Ridge Regression Results]")
    print(f"RMSE: {root_mean_squared_error(t_test, y_rr_sk)}")
    print(f"R2: {r2_score(t_test, y_rr_sk)}")
    print()

    # ---- Part (j) ---- #
    # Plot the results
    plt.figure(figsize=(12, 6))
    t_min, t_max = np.min(t_test), np.max(t_test)

    plt.subplot(2, 2, 1)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression")
    plt.scatter(t_test, y_lr)
    plt.plot([t_min, t_max], [t_min, t_max], "r-")

    plt.subplot(2, 2, 2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Ridge Regression")
    plt.scatter(t_test, y_rr)
    plt.plot([t_min, t_max], [t_min, t_max], "r-")

    plt.subplot(2, 2, 3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Scikit-learn Linear Regression")
    plt.scatter(t_test, y_lr_sk)
    plt.plot([t_min, t_max], [t_min, t_max], "r-")

    plt.subplot(2, 2, 4)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Scikit-learn Ridge Regression")
    plt.scatter(t_test, y_rr_sk)
    plt.plot([t_min, t_max], [t_min, t_max], "r-")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
