import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

RS = 1234
np.random.seed(RS)

# Part (a)
X_blob, t_blob = make_blobs(cluster_std=[1.0, 2.5, 0.5], random_state=RS)

# Part (b)
X_circle, t_circle = make_circles(
    factor=0.5,
    noise=0.05,
    random_state=RS,
)

# Part (c)
kmeans = KMeans(n_clusters=3, random_state=RS)
kmeans.fit(X_blob)
blob_cluster_centers = kmeans.cluster_centers_
blob_labels = kmeans.labels_
kmeans = KMeans(n_clusters=3, random_state=RS)
kmeans.fit(X_circle)
circle_cluster_centers = kmeans.cluster_centers_
circle_labels = kmeans.labels_

# Part (d)
gm = GaussianMixture(n_components=3, random_state=RS)
gm.fit(X_blob)
blob_gm_cluster_centers = gm.means_
blob_gm_cov = gm.covariances_
blob_gm_labels = gm.predict(X_blob)
gm = GaussianMixture(n_components=3, random_state=RS)
gm.fit(X_circle)
circle_gm_cluster_centers = gm.means_
circle_gm_cov = gm.covariances_
circle_gm_labels = gm.predict(X_circle)

# Part (e)
# _, axes = plt.subplots(1, 3, figsize=(15, 4))
# axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=t_blob, cmap="viridis")
# axes[0].set_title("Ground Truth Labels")
#
# axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=blob_labels, cmap="viridis")
# axes[1].scatter(
#     blob_cluster_centers[:, 0],
#     blob_cluster_centers[:, 1],
#     color="black",
#     marker="x",
#     s=64,
#     label="Centers",
# )
# axes[1].set_title("K-means Clustering")
# axes[1].legend()
#
# axes[2].scatter(X_blob[:, 0], X_blob[:, 1], c=blob_gm_labels, cmap="viridis")
# axes[2].scatter(
#     blob_gm_cluster_centers[:, 0],
#     blob_gm_cluster_centers[:, 1],
#     color="black",
#     marker="x",
#     s=64,
#     label="Centers",
# )
# axes[2].set_title("Gaussian Mixture Model")
# axes[2].legend()
#
# plt.tight_layout()
# plt.show()
#
# # Part (f)
# _, axes = plt.subplots(1, 3, figsize=(15, 4))
# axes[0].scatter(X_circle[:, 0], X_circle[:, 1], c=t_circle, cmap="viridis")
# axes[0].set_title("Ground Truth Labels")
#
# axes[1].scatter(X_circle[:, 0], X_circle[:, 1], c=circle_labels, cmap="viridis")
# axes[1].scatter(
#     circle_cluster_centers[:, 0],
#     circle_cluster_centers[:, 1],
#     color="black",
#     marker="x",
#     s=64,
#     label="Centers",
# )
# axes[1].set_title("K-means Clustering")
# axes[1].legend()
#
# axes[2].scatter(
#     X_circle[:, 0], X_circle[:, 1], c=circle_gm_labels, cmap="viridis"
# )
# axes[2].scatter(
#     circle_gm_cluster_centers[:, 0],
#     circle_gm_cluster_centers[:, 1],
#     color="black",
#     marker="x",
#     s=64,
#     label="Centers",
# )
# axes[2].set_title("Gaussian Mixture Model")
# axes[2].legend()
#
# plt.tight_layout()
# plt.show()


# Part (g)
def plot_gmm_ellipses(_ax, _gmm, _colors=None, _alpha=0.3):
    """
    Plots one ellipse per GMM cluster on a given Axes.

    Parameters:
        _ax: matplotlib Axes object
        _gmm: trained sklearn GaussianMixture object
        _colors: optional list of colors for each ellipse
        _alpha: transparency level of ellipses
    """
    for i, (mean, covar) in enumerate(zip(_gmm.means_, _gmm.covariances_)):
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(covar)

        # Sort eigenvalues/vecs from largest to smallest
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Width and height (2*sqrt(2*v) for each axis)
        width, height = 2 * np.sqrt(2 * vals)

        # Angle in degrees (from the largest eigenvector)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Choose color
        color = _colors[i] if _colors is not None else "black"

        # Create and add ellipse
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            color=color,
            alpha=_alpha,
        )
        _ax.add_patch(ellipse)


_, ax = plt.subplots()
ax.scatter(
    X_circle[:, 0], X_circle[:, 1], c=circle_gm_labels, s=20, cmap="viridis"
)
ax.scatter(
    circle_gm_cluster_centers[:, 0],
    circle_gm_cluster_centers[:, 1],
    marker="x",
    color="black",
)

# Plot GMM ellipses
plot_gmm_ellipses(ax, gm)

plt.title("GMM with Covariance Ellipses")
plt.axis("equal")
plt.show()
