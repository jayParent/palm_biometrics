import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from collections import Counter
import csv

# PCA n_components -- 64, 128 and 256 all have same results(90) -- 32(88)


def find_best_parameters(dataFile, labelsFile):
    with open(dataFile, 'rb') as fp:
        data = pickle.load(fp)
    with open(labelsFile, 'rb') as fp:
        labels = pickle.load(fp)

    pca = PCA(n_components=20)
    pca.fit(data)

    X = pca.fit_transform(data)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Set the parameters by cross-validation
    param_grid = [{'kernel': ['rbf', 'linear'], 'gamma': [0.015, 0.02, 0.025],
                   'C': [20, 25, 30]}]

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    grid.fit(X_train, y_train)

    print(grid.best_params_)
    print(grid.best_estimator_)



def get_cross_val_scores(dataFile, labelsFile):
    with open(f'{dataFile}', 'rb') as fp:
        data = pickle.load(fp)
    with open(f'{labelsFile}', 'rb') as fp:
        labels = pickle.load(fp)

    pca = PCA(n_components=20)
    pca.fit(data)

    X = pca.fit_transform(data)
    y = np.array(labels)
    y = y.astype(np.float64)

    clf = SVC(C=25, kernel='rbf', gamma=0.025)
    scores = cross_val_score(clf, X, y, cv=8)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def test_dataset_multiClass(dataFile, labelsFile):
    with open(f'{dataFile}', 'rb') as fp:
        data = pickle.load(fp)
    with open(f'{labelsFile}', 'rb') as fp:
        labels = pickle.load(fp)

    pca = PCA(n_components=20)
    pca.fit(data)

    X = pca.fit_transform(data)
    y = np.array(labels)
    y = y.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    clf = SVC(C=20, kernel='rbf', gamma=0.015)
    clf.fit(X_train, y_train)

    # true = 0
    # false = 0
    # predictions = clf.predict(X_test)

    # with open('multiclass_predictions.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(['Sujet', 'Prediction', 'Resultat'])
        
    #     for label, prediction in zip(y_test, predictions):
    #         if label == prediction:
    #             result = 'True'
    #             true += 1
    #         else:
    #             result = 'False'
    #             false += 1
    #         writer.writerow([label, prediction, result])
    # print(true, false)
        

    print(clf.score(X_test, y_test))
