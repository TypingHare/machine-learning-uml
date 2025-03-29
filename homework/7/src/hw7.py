import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_pairs
from sklearn import svm
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

RANDOM_STATE = 4710
np.random.seed(RANDOM_STATE)

# Part (a) - (b)
# Print the complete description of the dataset
lfw_pairs = fetch_lfw_pairs()
target_names = lfw_pairs.target_names
print(lfw_pairs.DESCR)

# Part (c)
lfw_pairs_train = fetch_lfw_pairs(subset="train")
lfw_pairs_test = fetch_lfw_pairs(subset="test")
X_train = lfw_pairs_train.data
X_test = lfw_pairs_test.data
t_train = lfw_pairs_train.target
t_test = lfw_pairs_test.target

# Part (d)
classifier = svm.SVC(kernel="rbf", class_weight="balanced")
classifier.fit(X_train, t_train)
t_pred = classifier.predict(X_test)
print(f"\n{'-' * 35} Part (d) {'-' * 35}")


def print_classification_report(_t_pred):
    """Prints the classification report with the given predictions."""
    print(classification_report(t_test, _t_pred, target_names=target_names))


def display_confusion_matrix(_t_pred):
    """Displays the confusion matrix with the given predictions."""
    labels = classifier.classes_
    cm = confusion_matrix(t_test, _t_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    fig.suptitle("Confusion Matrix - SVM Face Pairs")
    fig.tight_layout()
    plt.show()


print_classification_report(t_pred)
display_confusion_matrix(t_pred)


# Part (e)
# Commented when submit because it would take a lot of time (around 30 minutes)
# param_distributions = {
#     "C": uniform(loc=0.1, scale=10),
#     "gamma": uniform(loc=0.001, scale=1),
#     "kernel": ["linear", "sigmoid", "rbf"],
# }
# random_search = RandomizedSearchCV(
#     estimator=svm.SVC(class_weight="balanced"),
#     param_distributions=param_distributions,
#     n_iter=50,
#     cv=5,
#     verbose=1,
#     n_jobs=1,
#     random_state=42,
# )
# random_search.fit(X_train, t_train)
# print(f"\n{'-' * 35} Part (e) {'-' * 35}")
# print("Best params: ")
# print(random_search.best_params_)

# Part (f)
# Best parameters from part (e):
#     C = 2.087156815341724
#     gamma = 0.006522117123602399
#     kernel = 'rbf'
#
# best_classifier = random_search.best_estimator_
best_classifier = svm.SVC(
    kernel="rbf",
    class_weight="balanced",
    C=2.087156815341724,
    gamma=0.006522117123602399,
)
best_classifier.fit(X_train, t_train)
t_pred_best = best_classifier.predict(X_test)

print(f"\n{'-' * 35} Part (f) {'-' * 35}")
print_classification_report(t_pred_best)
display_confusion_matrix(t_pred_best)

# Part (g) and (h) is in the document
print(f"\n{'-' * 37} End {'-' * 38}")
