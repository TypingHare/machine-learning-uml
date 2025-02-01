import math

import numpy as np


if __name__ == "__main__":
    coefficients = np.zeros(16)
    coefficients = coefficients.reshape(16, 1)
    for i in range(0, 16):
        if i % 2 == 0:
            coefficients[i] = float(0)
        else:
            coefficients[i] = (-1) ** (i // 2) * (2 * np.pi) ** i / math.factorial(i)

    print(coefficients)
