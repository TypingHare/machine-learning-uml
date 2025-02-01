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
        :return: A matrix of transformed features.
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
        # The weight matrix
        self.w = None

    def _reset(self):
        self.w = None

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray):

        # This is the fit function. Implement the equations that calculate
        # the mean and variance of the solution
        # --- Your Code Here ---#
        # part (e)
        self.w = None

    def _predict(self, x: np.ndarray, return_std: bool = False):
        # --- Your Code Here ---#
        # part (f)
        y = x


def rmse(a, b):
    """Calculates the RMSE error between two vectors"""
    # --- Your Code Here ---#
    # part (j)
    return 0.0


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
    x_test_pf9 = pf3.transform(x_test)
    print(x_train_pf9)

    # part (g)
    # part (h)
    # part (i)

    # part (k)
    # part (l)


if __name__ == "__main__":
    print("--- Homework 1 ---")
    main()
