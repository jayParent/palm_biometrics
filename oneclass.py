import pickle
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import where
from dataset_import import import_data, filter_and_pca_subjects
from sklearn.svm import SVC, OneClassSVM


def create_classifiers(subjects):
    classifiers = []
    clf = OneClassSVM(kernel='linear')

    for subject in good_subjects:
        subject_clf = clf.fit(subject[0])
        classifiers.append([subject_clf, subject[1]])

    return classifiers


subjects = import_data('oneClass_data')
# print(subjects[0][0].shape, subjects[1][0].shape)

good_subjects = filter_and_pca_subjects(subjects, 12)
# print(good_subjects[0][0].shape, good_subjects[0][1])
# print(good_subjects[1][0].shape, good_subjects[1][1])

classifiers = create_classifiers(good_subjects)
# print(len(classifiers))

train_number = 0
clf_name = classifiers[train_number][1]
clf = classifiers[train_number][0]
train_data = good_subjects[train_number][0]

test_number = 0
test_subject_name = good_subjects[test_number][1]
test_subject_data = good_subjects[test_number][0]

pred = clf.predict(test_subject_data)

anom_index = where(pred == -1)
outliers = train_data[anom_index]

print(f'Fitted on: {clf_name}')
print(f'Prediction for: {test_subject_name}', pred)

plt.scatter(train_data[:, 0], train_data[:, 1])
plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
plt.show()
