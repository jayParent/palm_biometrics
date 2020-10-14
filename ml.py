import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from collections import Counter
import csv

# PCA n_components -- 64, 128 and 256 all have same results(90) -- 32(88)


def find_best_parameters(dataFile, labelsFile):

    with open(dataFile, 'rb') as fp:
        data = pickle.load(fp)
    with open(labelsFile, 'rb') as fp:
        labels = pickle.load(fp)

    data = np.array(data)
    if data.shape[0] < 128:
        n_components = data.shape[0]
    else:
        n_components = 128

    pca = PCA(n_components=n_components)
    pca.fit(data)

    X = pca.fit_transform(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Set the parameters by cross-validation
    param_grid = [{'kernel': ['rbf'], 'gamma': [0.015, 0.02, 0.025],
                   'C': [20, 25, 30]}]

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)

    # clf = SVC(C=20, kernel='rbf', gamma=0.02) -- Best for whole dataset


def get_cross_val_scores(dataFile, labelsFile):

    with open(dataFile, 'rb') as fp:
        data = pickle.load(fp)
    with open(labelsFile, 'rb') as fp:
        labels = pickle.load(fp)

    data = np.array(data)
    if data.shape[0] < 128:
        n_components = data.shape[0]
    else:
        n_components = 64

    pca = PCA(n_components=n_components)
    pca.fit(data)

    X = pca.fit_transform(data)
    y = np.array(labels)

    clf = SVC(C=20, kernel='rbf', gamma=0.02)
    scores = cross_val_score(clf, X, y, cv=8)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# # # Whole dataset -- Accuracy: 0.88 (+/- 0.11)
# get_cross_val_scores('data.txt', 'labels.txt')

# # # Testing different number of subjects -- 2 -- Accuracy: 0.94 (+/- 0.22)
# get_cross_val_scores('data_2_subjects.txt', 'labels_2_subjects.txt')

# # # Testing different number of subjects -- 10 -- Accuracy: 0.88 (+/- 0.17)
# get_cross_val_scores('data_10_subjects.txt', 'labels_10_subjects.txt')

with open('data.txt', 'rb') as fp:
    data = pickle.load(fp)
with open('labels.txt', 'rb') as fp:
    labels = pickle.load(fp)

pca = PCA(n_components=128)
pca.fit(data)

X = pca.fit_transform(data)
y = np.array(labels)
y = y.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = SVC(C=20, kernel='rbf', gamma=0.02)
clf.fit(X_train, y_train)

final_predictions = clf.predict(X_test)

errors = []

for p, l in zip(final_predictions, y_test):
    if p != l:
        errors.append(l)

print(Counter(errors))
print(len(errors))
