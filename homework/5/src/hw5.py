import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 1234


def random_digit_plot(_size: tuple[int, int], _data, _target, _pred=None):
    """Generates subplots of random digits from the digits dataset."""
    rows, cols = _size
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
    indices = np.random.choice(len(_data), size=n, replace=False)

    caption = "Training: " if _pred is None else "Testing: "
    for subplot, idx in zip(axes.ravel(), indices):
        image = _data[idx]
        subplot.imshow(image, cmap="gray_r", interpolation="nearest")
        is_correct = _pred is None or _pred[idx] == _target[idx]
        color = "blue" if is_correct else "red"
        subplot.set_title(f"{caption}{_target[idx]}", color=color)
        subplot.axis("off")

    plt.tight_layout()
    plt.show()


# Part (a)
digits = datasets.load_digits()
print("-" * 36 + "Part (a)" + "-" * 36)
print("The number of samples in the digits dataset:", len(digits.images))
print("The shape of each image:", digits.images[0].shape)

# Part (b)
random_digit_plot((4, 4), digits.images, digits.target)

# Part (c)
X = digits.images.reshape(len(digits.images), -1)

# Part (d)
X_scaled = MinMaxScaler().fit_transform(X)
t = digits.target
X_train, X_test, t_train, t_test, _, idx_test = train_test_split(
    X_scaled,
    t,
    np.arange(len(X_scaled)),
    test_size=0.3,
    random_state=RANDOM_STATE,
)
print("-" * 36 + "Part (d)" + "-" * 36)
print("Size of training set:", len(X_train))
print("Size of test set:", len(X_test))

# Part (e)
mlp = MLPClassifier(
    max_iter=10_000, hidden_layer_sizes=(54, 55, 58), alpha=1e-6
)  # Refer to the PDF for the reason of using these parameters
mlp.fit(X_train, t_train)

# Part (f)
y_pred = mlp.predict(X_test)
print("-" * 36 + "Part (f)" + "-" * 36)
print(confusion_matrix(t_test, y_pred))

# Part (g)
images_test = digits.images[idx_test]
random_digit_plot((4, 4), images_test, y_pred, t_test)


# Part (h)
def train_and_test(
    _mlp: MLPClassifier, _output: bool = True, _random_state=RANDOM_STATE
) -> float:
    _mlp.random_state = _random_state
    _mlp.fit(X_train, t_train)
    cm = confusion_matrix(t_test, _mlp.predict(X_test))
    off_diagonal_sum = np.sum(cm) - np.sum(np.diagonal(cm))
    _accuracy = 1 - off_diagonal_sum / len(X_train)

    if _output:
        print(
            f"{_mlp.hidden_layer_sizes}; {_mlp.activation}; {_mlp.max_iter}; {_mlp.alpha} => accuracy: {_accuracy.round(6)}"
        )

    return _accuracy


print("-" * 36 + "Part (h)" + "-" * 36)
# train_and_test(MLPClassifier(max_iter=1_000))
# train_and_test(MLPClassifier(max_iter=1_000, activation="tanh"))
# train_and_test(MLPClassifier(max_iter=1_000, activation="logistic"))
# train_and_test(MLPClassifier(max_iter=1_000, hidden_layer_sizes=(128, 64)))
# train_and_test(MLPClassifier(max_iter=1_000, hidden_layer_sizes=(128, 64, 32)))
# train_and_test(MLPClassifier(max_iter=2_000, alpha=1e-6))
# train_and_test(MLPClassifier(max_iter=10_000, alpha=1e-8))
# train_and_test(MLPClassifier(max_iter=1_000, solver="sgd"))
# train_and_test(MLPClassifier(max_iter=1_000, solver="lbfgs"))

# Output
# (100,); relu; 1000; 0.0001 => accuracy: 0.988067
# (100,); tanh; 1000; 0.0001 => accuracy: 0.98568
# (100,); logistic; 1000; 0.0001 => accuracy: 0.986476
# (128, 64); relu; 1000; 0.0001 => accuracy: 0.990453
# (128, 64, 32); relu; 1000; 0.0001 => accuracy: 0.989658
# (100,); relu; 2000; 1e-06 => accuracy: 0.988067
# (100,); relu; 10000; 1e-08 => accuracy: 0.988067
# (100,); relu; 1000; 0.0001 => accuracy: 0.984885
# (100,); relu; 1000; 0.0001 => accuracy: 0.979316


# Part (i)
print("-" * 36 + "Part (i)" + "-" * 36)
#
#
# def record_time(_fn: callable) -> tuple[float, any]:
#     start_time = time.time()
#     result = _fn()
#     return result, (time.time() - start_time) * 1000
#
#
# best_accuracy = 0.0
# hidden_layer_sizes_with_best_accuracy = (1, 1, 1)
# best_mul = 1000000000.0
# hidden_layer_sizes_with_best_mul = (1, 1, 1)
# layer_range_min = 70
# layer_range_max = 80
# random_state = 10086
#
# for first_layer in range(layer_range_min, layer_range_max):
#     for second_layer in range(layer_range_min, layer_range_max):
#         for third_layer in range(layer_range_min, layer_range_max):
#             hidden_layer_sizes = (
#                 first_layer,
#                 second_layer,
#                 third_layer,
#             )
#
#             def fn():
#                 return train_and_test(
#                     MLPClassifier(
#                         max_iter=10_000,
#                         alpha=1e-6,
#                         hidden_layer_sizes=hidden_layer_sizes,
#                     ),
#                     _output=False,
#                     _random_state=random_state,
#                 )
#
#             accuracy, duration = record_time(fn)
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 hidden_layer_sizes_with_best_accuracy = hidden_layer_sizes
#
#             mul = (1 - accuracy) * duration
#             if mul < best_mul:
#                 best_mul = mul
#                 hidden_layer_sizes_with_best_mul = hidden_layer_sizes
#
#             print(
#                 f"{str(hidden_layer_sizes):<15} {accuracy:<10.6f} {duration:<10.6f} {mul:<10.6f}"
#             )
#
# print(best_accuracy, hidden_layer_sizes_with_best_accuracy)
# print(best_mul, hidden_layer_sizes_with_best_mul)
