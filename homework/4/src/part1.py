"""
This is Homework 4 in COMP4220-Machine Learning
University of Massachusetts Lowell
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import functools


def create_synthetic_data(add_outliers=False, add_class=False):
    """This function creates a synthetic 2D data set for classification.
    It can be used to add outliers and well as more classes."""
    x0 = np.random.normal(size=50).reshape(-1, 2) - 1
    x1 = np.random.normal(size=50).reshape(-1, 2) + 1

    if add_outliers:
        x2 = np.random.normal(size=10).reshape(-1, 2) + np.array([5.0, 10.0])
        return np.concatenate([x0, x1, x2]), np.concatenate(
            [np.zeros(25), np.ones(30)]
        ).astype(int)

    if add_class:
        x2 = np.random.normal(size=50).reshape(-1, 2) + 3.0
        return np.concatenate([x0, x1, x2]), np.concatenate(
            [np.zeros(25), np.ones(25), 2 + np.zeros(25)]
        ).astype(int)

    return np.concatenate([x0, x1]), np.concatenate(
        [np.zeros(25), np.ones(25)]
    ).astype(int)


class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]  # ensure size (:,1) instead of (:,)
        x_t = x.transpose()  # return the data to its initial shape
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda _x, y: _x * y, items))
        return np.asarray(features).transpose()


class Classifier(object):
    pass


class LogisticRegression(Classifier):
    def __init__(self):
        self.w = None

    @staticmethod
    def _sigmoid(a):
        """Sigmoid function is defined as σ(a) = 1 / (1 + e^{-a})."""
        a = np.clip(a, -1e2, 1e2)
        return 1 / (1 + np.exp(-a))

    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 100
    ) -> None:
        # Add bias term
        x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])

        n_features = x_train.shape[1]
        w = np.zeros(n_features)
        for _ in range(max_iter):
            # Previous w
            prev_w = w.copy()

            # Prediction
            y = self._sigmoid(x_train @ w)

            # Gradient of the log-likelihood
            gradient = x_train.T @ (y - y_train)

            # Hessian matrix
            hessian = x_train.T @ np.diagflat(y * (1 - y)) @ x_train

            # Newton-Raphson update: w = w - H^{-1} * gradient
            w -= np.linalg.pinv(hessian) @ gradient

            # Check for convergence
            if np.all(np.abs(w - prev_w) < 1e-6):
                break

        self.w = w

    def proba(self, x: np.ndarray) -> np.ndarray:
        # Add bias term for x
        x = np.hstack([np.ones((x.shape[0], 1)), x])
        return self._sigmoid(x @ self.w)

    def classify(self, x: np.ndarray, threshold: float = 0.5):
        p = self.proba(x)
        return np.where(p < threshold, 0, 1)


# noinspection PyPep8Naming
def test_LogisticRegression():
    # part (h)
    x_train, t_train = create_synthetic_data()

    # part (i)
    x_min, x_max, y_min, y_max = -5, 15, -5, 15
    x1_test = np.linspace(x_min, x_max, 100)
    x2_test = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x1_test, x2_test)

    # These are grid points covering all possible combinations of x and y in
    # [-5, 15]
    x_test = np.column_stack((xx.ravel(), yy.ravel()))

    # part (j)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train, t_train)

    # part (k)
    y_lr = logistic_regression.classify(x_test)

    # part (l)
    plt.figure(figsize=(10, 8))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Logistic Regression")

    # Plot the training data
    # Scatter the training samples (class 0 in red; class 1 in blue)
    plt.scatter(
        x_train[t_train == 0, 0],
        x_train[t_train == 0, 1],
        color="r",
        label="Class 0 (Train)",
    )
    plt.scatter(
        x_train[t_train == 1, 0],
        x_train[t_train == 1, 1],
        color="b",
        label="Class 1 (Train)",
    )

    # Plot the contour
    z = y_lr.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.5, cmap="RdBu")

    plt.show()


if __name__ == "__main__":
    test_LogisticRegression()
