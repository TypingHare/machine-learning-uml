"""
This is Homework 6 in COMP4220-Machine Learning 
University of Massachusetts Lowell
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    ExpSineSquared,
    Matern,
)


def generate_training_dataset():
    x = np.random.uniform(0, 5, size=(10, 1))
    t = np.sin((x - 2.5) ** 2).flatten()

    return x, t


def regression_with_noise_free_data():
    # --- part a ---
    x, t = generate_training_dataset()

    # --- part b ---
    kernel = RBF(length_scale_bounds=(0.01, 100))

    # --- part c ---
    gpr = GaussianProcessRegressor(kernel)

    # --- part d ---
    x_test = np.linspace(0, 5, 100).reshape(-1, 1)

    # --- part e, f, g --- (modeling)
    # Get prior samples
    y_mean_prior, y_std_prior = gpr.predict(x_test, return_std=True)
    y_samples_prior = gpr.sample_y(x_test, 5)

    # Fit with observations (training set) and get posterior samples
    gpr.fit(x, t)
    y_mean_post, y_std_post = gpr.predict(x_test, return_std=True)
    y_samples_post = gpr.sample_y(x_test, 5)

    # --- part e, f, g --- (plot)
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    fig.suptitle("Gaussian process using RBF kernel")
    axs[0].set_xlim([-0.5, 5.5])
    axs[0].set_ylim([-3.0, 3.0])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Samples from prior")
    axs[0].plot(x_test, y_mean_prior, "b-", label="m_prior")
    axs[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    axs[0].fill_between(
        x_test.ravel(),
        y_mean_prior - y_std_prior,
        y_mean_prior + y_std_prior,
        alpha=0.2,
        color="gray",
        label="m_prior",
    )
    for i in range(5):
        axs[0].plot(x_test, y_samples_prior[:, i], lw=1, ls="--")

    axs[1].set_xlim([-0.5, 5.5])
    axs[1].set_ylim([-3.0, 3.0])
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("Samples from posterior")
    axs[1].plot(x_test, y_mean_post, "b-", label="m_post")
    axs[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    axs[1].fill_between(
        x_test.ravel(),
        y_mean_post - y_std_post,
        y_mean_post + y_std_post,
        alpha=0.2,
        color="gray",
        label=r"m_post",
    )
    for i in range(5):
        axs[1].plot(x_test, y_samples_post[:, i], lw=1, ls="--")

    plt.tight_layout()
    plt.show()

    # --- part h ---
    print(f"------- Kernel Hyper-parameters --------")
    print(f"Before fit: {gpr.kernel}")
    print(f"After fit: {gpr.kernel_}")
    print(f"Log Marginal Likelihood: {gpr.log_marginal_likelihood_value_}")


def regression_with_noisy_data():
    # --- part i ---
    x, t = generate_training_dataset()
    t_noisy = t + np.random.normal(0, 0.2, t.shape)
    x_test = np.linspace(0, 5, 100).reshape(-1, 1)

    # --- part j, k, l ---
    kernel_names = ["RBF", "RQ", "ExpSinSquared", "Matern"]
    kernels = [
        RBF(),
        RationalQuadratic(),
        ExpSineSquared(),
        Matern(),
    ]

    # Create 2x2 plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle("Gaussian Process Regression on noisy data")
    axs = axs.flatten()
    log_likelihoods = []

    for i, (kernel, kernel_name) in enumerate(zip(kernels, kernel_names)):
        gpr = GaussianProcessRegressor(kernel)
        gpr.fit(x, t_noisy)
        y_mean, y_std = gpr.predict(x_test, return_std=True)
        y_samples = gpr.sample_y(x_test, 5)
        log_likelihoods.append(gpr.log_marginal_likelihood_value_)

        axs[i].set_xlim([-0.5, 5.5])
        axs[i].set_ylim([-3.0, 3.0])
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].set_title(f"{kernel_name} Kernel")
        axs[i].plot(x_test, y_mean, "b-")
        axs[i].fill_between(
            x_test.ravel(),
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.2,
            color="gray",
        )
        for j in range(5):
            axs[i].plot(x_test, y_samples[:, j], lw=1, ls="--")

        print(f"\n------- {kernel_name} Kernel --------")
        print(f"Before fit: {kernel}")
        print(f"After fit: {gpr.kernel_}")
        print(f"Log Marginal Likelihood: {gpr.log_marginal_likelihood_value_}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # For consistency, we use a specific random seed throughout this question.
    # Please do not modify this seed.
    np.random.seed(4)

    regression_with_noise_free_data()
    regression_with_noisy_data()
