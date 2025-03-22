'''
This is Homework 6 in COMP4220-Machine Learning 
University of Massachusetts Lowell
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, Matern




def generate_training_dataset():
    # --- part a ---
    x = []
    t = []
    return x, t


def regression_with_noise_free_data():
    # --- part a ---
    x, t = generate_training_dataset()

    # --- part b ---
    kernel = RBF()

    # --- part c ---
    gpr = GaussianProcessRegressor()

    # --- part d ---
    x_test = []

    # --- part e, f, g --- (modeling)
    # get prior samples
    y_mean_prior, y_std_prior = [], []

    y_samples_prior = []
    # get posterior samples

    y_mean_post, y_std_post = [], []
    y_samples_post = []

    # --- part e, f, g --- (plot)
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))
    axs[0].set_xlim([-0.5,5.5])
    axs[0].set_ylim([-3.,3.])
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Samples from prior')

    axs[1].set_xlim([-0.5,5.5])
    axs[1].set_ylim([-3.,3.])
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    axs[1].set_title('Samples from posterior')
    fig.suptitle('Gaussian process using RBF kernel')
    plt.show()

    #--- part h ---
    print(f'------- Kernel Hyper-parameters --------')
    print(f'Before fit: {0}')
    print(f'After fit: {0}')
    print(f'Log Marginal Likelihood: {0}')


def regression_with_noisy_data():
    # --- part i ---
    x, t = generate_training_dataset()
    t_noisy = []
    x_test = []

    # --- part j, k, l ---
    Kernel_names = ['RBF', 'RQ', 'ExpSinSquared', 'Matern']
    gpr = GaussianProcessRegressor()
    y_mean, y_std = [], []
    y_samples = []

    print(f'------- Kernel Hyper-parameters --------')
    print(f'Before fit: {0}')
    print(f'After fit: {0}')
    print(f'Log Marginal Likelihood: {0}')

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,6))
    fig.suptitle('Gaussian Process Regression on noisy data')
    plt.show()


if __name__=='__main__':
    # For consistency, we use a specific random seed throughout this question.
    # Please do not modify this seed.
    np.random.seed(4)
    
    regression_with_noise_free_data()
    regression_with_noisy_data()
    # It would be best to not run these functions sequentially
    # instead comment out one and run the other one, or develop in two separate files.
    # this is to reduce the risk of shared variables between functions and other events
    # that can cause the results from the first part influence the result of the second part.
