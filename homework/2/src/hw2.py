import numpy as np
import matplotlib.pyplot as plt


def gather_uniform_samples(n: int) -> np.ndarray:
    """Generates N samples from uniform distribution U[0, 1].

    Args:
        n: Number of samples to generate.

    Returns:
        A vector of N samples from U[0, 1].
    """
    return np.random.uniform(low=0.0, high=1.0, size=n)


def plot_histograms(all_samples: np.ndarray, plot_indices: list[int]) -> None:
    """Plot histograms for specified sample sets.

    Args:
        all_samples: Array of shape (m, n) containing all sample sets.
        plot_indices: List of indices to plot.
    """
    fig, axes = plt.subplots(1, len(plot_indices), figsize=(15, 4))
    fig.suptitle("Histogram of Samples")

    # Plot each histogram
    for i, idx in enumerate(plot_indices):
        axes[i].hist(all_samples[idx], bins=64, density=True)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 5)
        axes[i].set_title(f"$s_{{{idx + 1}}}$")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def main():
    # The number of samples that are gathered in each sample set
    n = 10_000

    # The number of sample sets
    m = 10

    # All the sample sets
    sample_sets = np.zeros((m, n))
    for i in range(m):
        sample_sets[i] = gather_uniform_samples(n)

    # The accumulation of samples
    s = np.zeros((m, n))
    for i in range(m):
        s[i] = np.average(sample_sets[: (i + 1)], axis=0)

    plot_histograms(s, [0, 1, 9])


if __name__ == "__main__":
    main()
