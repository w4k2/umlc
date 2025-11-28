import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RANDOM_STATE = 90210

SPLITS_DIR = "_splits"
PREDICTIONS_DIR = "_results"
REPORTS_DIR = "_reports"

SPLITS = RepeatedStratifiedKFold(random_state=RANDOM_STATE)
N_SPLITS = SPLITS.get_n_splits()

METHODS = {
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "CRT": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(random_state=RANDOM_STATE),
    "MLP": MLPClassifier(random_state=RANDOM_STATE)
}

DATASETS = {
    "madelon-0.5": make_classification(flip_y=0, weights=[0.5, 0.5], random_state=RANDOM_STATE),
    "madelon-0.4": make_classification(flip_y=0, weights=[0.6, 0.4], random_state=RANDOM_STATE),
    "madelon-0.3": make_classification(flip_y=0, weights=[0.7, 0.3], random_state=RANDOM_STATE),
    "madelon-0.2": make_classification(flip_y=0, weights=[0.8, 0.2], random_state=RANDOM_STATE),
    "madelon-0.1": make_classification(flip_y=0, weights=[0.9, 0.1], random_state=RANDOM_STATE),
}

METRICS = {
    "BAC": balanced_accuracy_score,
    "AUC": roc_auc_score,
}