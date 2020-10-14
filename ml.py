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
from collections import Counter


with open('labels.txt', 'rb') as fp:
    labels = pickle.load(fp)
with open('pca_data.txt', 'rb') as fp:
    X = pickle.load(fp)

y = np.array(labels)

# Discard subjects with < 8 good ROI
label_count = Counter(labels)
print(label_count)

# Number of ROIs
print(len(labels))

# Number of subjects
print(len(label_count))

# Principal Component Analysis
# pca = PCA(n_components=128)
# pca.fit(data)


def find_best_parameters():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Set the parameters by cross-validation
    param_grid = [{'kernel': ['rbf'], 'gamma': [0.015, 0.02, 0.025],
                   'C': [20, 25, 30]}]

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)

    # clf = SVC(C=20, kernel='rbf', gamma=0.02)


def get_cross_val_scores(C, kernel, gamma):
    clf = SVC(C=C, kernel=kernel, gamma=gamma)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


get_cross_val_scores(20, 'rbf', 0.02)

# Testing different number of subjects -- 2
# with open('labels_2_subjects.txt', 'rb') as fp:
#     labels = pickle.load(fp)
# with open('data_2_subjects.txt', 'rb') as fp:
#     X = pickle.load(fp)

# X = np.array(X)
# y = np.array(labels)
# pca = PCA(n_components=32)
# pca.fit(X)

# print(X.shape, y.shape)
