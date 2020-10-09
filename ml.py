import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
from numpy import load
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA

with open('data.txt', 'rb') as fp:
    data = pickle.load(fp)
with open('labels.txt', 'rb') as fp:
    labels = pickle.load(fp)

data = np.array(data)
y = np.array(labels)

pca = PCA(n_components=150)
pca.fit(data)

X = pca.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = SVC(kernel='poly', degree=4, coef0=1, C=5)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
