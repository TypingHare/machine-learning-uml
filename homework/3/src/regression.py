# Linear Regression models
# COMP.4220 Machine Learning

import numpy as np


class Regression(object):
    def __init__(self):
        pass
        # self.w = None
        # self.t_mean = 0


class LinearRegression(Regression):
    """Linear regression"""

    def __init__(self):
        self.w = None
        self.var = None

    def fit(self, x_train: np.ndarray, t_train: np.ndarray):
        """Perform the least squares fitting.
        Parameters
        ----------
        x_train : np.ndarray
            training independent variable `(N, D)`
        t_train : np.ndarray
            training dependent variable `(N, )`
        """
        self.w = np.linalg.pinv(x_train) @ t_train
        self.var = np.mean(np.square(x_train @ self.w - t_train))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """Return prediction given input.
        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)
        return_std : bool, optional
            returns standard deviation of each prediction if True

        Returns
        -------
        y : np.ndarray
            prediction of each sample `(N, )`
        y_std : np.ndarray
            standard deviation of each prediction `(N, )`
        """
        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y


class RidgeRegression(Regression):
    """Ridge regression"""

    def __init__(self, _lambda: float = 1.0):
        """
        Parameters
        ----------
        _lambda : float, optional
            Regularization Coefficient
        """
        self._lambda = _lambda
        self.w = None

    def fit(self, x_train: np.ndarray, t_train: np.ndarray):
        """Maximum A Posteriori (MAP) estimation.
        Parameters
        ----------
        x_train : np.ndarray
            training data independent variable (N, D)
        t_train : np.ndarray
            training data dependent variable (N,)
        """
        # textbook (3.28)
        # (X^TX + λI)w = X^Tt
        # Solve for w, we have: w = (X^T X + λI)^(-1) X^T t
        # Here, A = (X^TX + λI); B = X^Tt
        D = x_train.shape[1]
        A = (x_train.T @ x_train) + self._lambda * np.eye(D)
        B = x_train.T @ t_train
        self.w = np.linalg.solve(A, B)

    def predict(self, x: np.ndarray):
        """Return prediction.
        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)

        Returns
        -------
        np.ndarray
            prediction of each input `(N, )`
        """
        return x @ self.w
