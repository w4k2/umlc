import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from random_classifier import RandomClassifier
from sklearn.neighbors import KNeighborsClassifier

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

DATASETS = [
    load_breast_cancer(return_X_y=True),
    load_iris(return_X_y=True)
]

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=15),
    RandomClassifier(random_state=1000),
]

scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

for dataset_idx, (X, y) in enumerate(DATASETS):
    for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(clf_prot)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score = accuracy_score(y[test], y_pred)
            scores[dataset_idx, classifier_idx, fold_idx] = score
        
np.save("scores", scores)