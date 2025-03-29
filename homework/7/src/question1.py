"""
This module plots the function

    f(x) = 8x_1^2 - 2x_2

where x = (x_1, x_2)^T; and find the minimal value of f(x) with the following
constraint:

    x_1^2 + x^2 = 1

The detailed process is in the document. The minimal point is f((0, 1)^T) = -2.
"""

import numpy as np
import matplotlib.pyplot as plt

# Create a canvass
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface of f(x) = 8x_1^2 - 2x_2
x1, x2 = np.meshgrid(np.linspace(-2, 2, 256), np.linspace(-2, 2, 256))
f = 8 * x1**2 - 2 * x2
ax.plot_surface(x1, x2, f, alpha=0.6, cmap="viridis", edgecolor="none")

# Plot the constraint circle in the plane
# The constraint is x_1^2 + x^2 = 1
theta = np.linspace(0, 2 * np.pi, 300)
cx = np.cos(theta)
cy = np.sin(theta)
cz = 8 * cx**2 - 2 * cy
ax.plot(cx, cy, cz, label="Constraint $x_1^2 + x_2^2 = 1$")

# Plot the minimum point (0, 1, -2)
ax.scatter(0, 1, -2, color="black", s=60, label="Minimum Point $(0, 1, -2)$")

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")
ax.set_title("Lagrange multipliers")
ax.legend()

plt.tight_layout()
plt.show()
