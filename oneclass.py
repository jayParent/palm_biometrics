import pickle
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import random
from numpy import where, quantile
from dataset_import import import_data, filter_and_pca_subjects
from sklearn.svm import SVC, OneClassSVM

data_filename = 'oneClass_data_oneHand'

def create_classifiers(subjects):
    classifiers = []
    clf = OneClassSVM(kernel='linear')

    for subject in subjects:
        subject_clf = clf.fit(subject[0])
        classifiers.append([subject_clf, subject[1]])

    return classifiers


def test_dataset(subjects, classifiers):
    for classifier, subject in zip(classifiers, subjects):
        subject_number = subject[1]
        rnd_subject_test = random.randint(0, len(subjects) - 1)

        clf = classifier[0]
        X = subject[0]
        outliers = subjects[rnd_subject_test][0]
        test = np.concatenate((X, outliers), axis=0)

        pred = clf.predict(test)

        print(f'Fitted on: {subject_number}')
        print(f'Prediction for: {subject_number} and {subjects[rnd_subject_test][1]}', pred)
        print('##############################################################################\n')

        # anom_index = where(pred == -1)
        # values = test[anom_index]

        # plt.scatter(test[:, 0], test[:, 1])
        # plt.scatter(values[:, 0], values[:, 1], color='r')
        # plt.show()

# subjects = import_data(data_filename)
# good_subjects = filter_and_pca_subjects(subjects, 8)
# classifiers = create_classifiers(good_subjects)

# test_dataset(good_subjects, classifiers)

