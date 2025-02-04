"""
This is Homework 1 in COMP4220-Machine Learning 
University of Massachusetts Lowell
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools, functools

# for reproducibility
np.random.seed(1234)


def generate_synthetic_dataset(_func, sample_size, std):
    """Generates 1D synthetic data for regression
    Inputs:
        func: a function representing the curve to sample from
        sample_size: number of points to generate
        std: standard deviation for additional noise
    Returns:
        A tuple (x, t), where x is a vector of independent variables, and t is
        a vector of target values.
    """
    x = np.linspace(0, 1, sample_size)
    t = _func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def func(x):
    """Represents the function f(x) = sin(2πx)"""
    return np.sin(2 * np.pi * x)


class PolynomialFeature(object):
    """Class for generating and transforming polynomial features"""

    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        """
        Example: given x = [3], degree = 4, the following polynomial feature
        vector will be returned: [[1], [3], [9], [27]]
        Example: given x = [5, 6], degree = 2, the following polynomial feature
        vector will be returned: [[1, 1], [5, 6]]
        :param x: A vector of features to be transformed to polynomial features.
        The size of the vector is N, where N is the number of samples.
        :return: A matrix of transformed features. The shape of the matrix is
        (N, M + 1), where N is the number of samples and M is the degree of the
        polynomial feature.
        """
        if x.ndim == 1:
            x = x[:, None]  # ensure size (:,1) instead of (:,)
        x_t = x.transpose()  # return the data to its initial shape
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda w, y: w * y, items))
        return np.asarray(features).transpose()


class Regression(object):
    """Basic class for the regression algorithms"""

    pass


class LinearRegression(Regression):
    def __init__(self):
        # The vector of weights; the length of this vector is M + 1, where M is
        # the degree of the polynomial feature.
        self.w = None

    def _reset(self):
        self.w = None

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the linear regression model using the normal equation.
        :param x_train: A matrix of shape (N, M + 1), where N is the number of
        samples, and M is the degree of the polynomial feature.
        :param y_train: A vector of length N, where N is the number of samples.
        :return: The product of the pseudo inverse of x_train and y_train. It
        is a vector of length M + 1, where M is the degree of the polynomial
        feature.
        """
        self.w = np.linalg.pinv(x_train) @ y_train

    def _predict(self, x: np.ndarray, return_std: bool = False):
        """
        Predicts the target values for the given input vector.
        :param x: The input matrix of shape (N, M).
        :param return_std: I don't know what it means.
        :return: A vector of length N, where N is the number of samples. (y)
        """
        return x @ self.w


def rmse(a, b):
    """Calculates the root-mean-square error between two vectors"""
    assert len(a) == len(b)
    return np.sqrt(np.mean((a - b) ** 2))


def draw_polynomial(x_test, y_test, weights):
    """Draws a plot of a polynomial."""

    # Create the plot of `func` in (0, 1)
    x = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 6))
    plt.plot(x, func(x), "g-", label="t = sin(2πx)")

    # Draw the test points
    for i in range(len(x_test)):
        plt.plot(x_test[i], y_test[i], "bo", markersize=7, markerfacecolor="none")

    # Draw the polynomial
    plt.plot(x, np.polyval(weights[::-1], x), "r-", label="fitting curve")

    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.show()


def draw_rmse(n, e_rms_train, e_rms_test):
    x = np.linspace(0, n - 1, n)
    plt.figure(figsize=(8, 6))

    # Lines
    plt.plot(x, e_rms_train, "b-", label="Training")
    plt.plot(x, e_rms_test, "r-", label="Testing")

    # Points
    for i in range(n):
        plt.plot(i, e_rms_train[i], "bo", markersize=7, markerfacecolor="none")
        plt.plot(i, e_rms_test[i], "ro", markersize=7, markerfacecolor="none")

    plt.xlabel("x")
    plt.ylabel("E_RMS")
    plt.legend()
    plt.show()


# noinspection PyProtectedMember
def main():
    # --- Your Code Here ---#
    # part (b) - generate training set
    x_train, t_train = generate_synthetic_dataset(func, 10, 0.25)

    # part (c) - generate test set
    x_test, t_test = generate_synthetic_dataset(func, 100, 0)

    # part (d)
    pf0 = PolynomialFeature(0)
    pf1 = PolynomialFeature(1)
    pf3 = PolynomialFeature(3)
    pf9 = PolynomialFeature(9)
    x_train_pf0 = pf0.transform(x_train)
    x_train_pf1 = pf1.transform(x_train)
    x_train_pf3 = pf3.transform(x_train)
    x_train_pf9 = pf9.transform(x_train)
    x_test_pf0 = pf0.transform(x_test)
    x_test_pf1 = pf1.transform(x_test)
    x_test_pf3 = pf3.transform(x_test)
    x_test_pf9 = pf9.transform(x_test)

    # part (g) - Train the four models with training sets
    lr0 = LinearRegression()
    lr0._fit(x_train_pf0, t_train)
    lr1 = LinearRegression()
    lr1._fit(x_train_pf1, t_train)
    lr3 = LinearRegression()
    lr3._fit(x_train_pf3, t_train)
    lr9 = LinearRegression()
    lr9._fit(x_train_pf9, t_train)

    # part (h) - Predict the testing targets
    t_test_pf0 = lr0._predict(x_test_pf0)
    t_test_pf1 = lr1._predict(x_test_pf1)
    t_test_pf3 = lr3._predict(x_test_pf3)
    t_test_pf9 = lr9._predict(x_test_pf9)

    # part (i)
    # draw_polynomial(x_train, t_train, lr0.w)
    # draw_polynomial(x_train, t_train, lr1.w)
    # draw_polynomial(x_train, t_train, lr3.w)
    # draw_polynomial(x_train, t_train, lr9.w)

    # part (j)
    print(f"RMSE when M = 0: {rmse(t_test, t_test_pf0)}")
    print(f"RMSE when M = 1: {rmse(t_test, t_test_pf1)}")
    print(f"RMSE when M = 3: {rmse(t_test, t_test_pf3)}")
    print(f"RMSE when M = 9: {rmse(t_test, t_test_pf9)}")

    # part (k) - loops over all orders in [0, 9]
    e_rms_train = []
    e_rms_test = []
    weights_vector = []
    for order in range(10):
        pf = PolynomialFeature(order)
        x_train_pf = pf.transform(x_train)
        x_test_pf = pf.transform(x_test)
        lr = LinearRegression()
        lr._fit(x_train_pf, t_train)
        t_train_pf = lr._predict(x_train_pf)
        t_test_pf = lr._predict(x_test_pf)
        e_rms_train.append(rmse(t_train, t_train_pf))
        e_rms_test.append(rmse(t_test, t_test_pf))
        weights_vector.append(lr.w)
    # draw_rmse(10, e_rms_train, e_rms_test)

    # part (l)
    print_weights(weights_vector)


def print_weights(weights_vector):
    # Pre-process weights
    _weights_vector = []
    for i in range(len(weights_vector)):
        _weights_vector.append([])
        for j in range(len(weights_vector[i])):
            _weights_vector[i].append(str(round(weights_vector[i][j], 2)))
            if _weights_vector[i][j][0] != "-":
                _weights_vector[i][j] = " " + _weights_vector[i][j]

    # Find the max widths of each column
    max_columns = [0] * len(_weights_vector)
    for i in range(len(_weights_vector)):
        for weight in _weights_vector[i]:
            max_columns[i] = max(max_columns[i], len(str(weight)))

    # Print the header
    print("\n[Polynomial Coefficients (Weights) Table]\n")
    print("w/M", end=" " * 4)
    for col in range(len(_weights_vector)):
        col_str = str(col)
        print(col_str, end=" " * (max_columns[col] + 2 - len(col_str)))
    print()
    print("-" * 100)

    # Print the table body
    rows = len(_weights_vector[-1])
    for row in range(rows):
        print(f"w*{row}", end=" " * 2)
        for col in range(len(_weights_vector)):
            if row < len(_weights_vector[col]):
                num_str = str(_weights_vector[col][row])
                print(num_str, end=" " * (max_columns[col] - len(num_str) + 2))
            else:
                print(end=" " * (max_columns[col] + 2))
        print()


if __name__ == "__main__":
    print("--- Homework 1 ---")
    main()
