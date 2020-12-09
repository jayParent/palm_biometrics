# -*- coding: UTF-8 -*-

from PROIE import *
from functions import *
from skimage import io
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import os
import pickle


def create_dataset_folders(sourceFolder, targetFolder, datasetFolder, clf):
    files = os.listdir(sourceFolder)

    if clf == 'oneClass':
        if not os.path.exists(targetFolder):
            os.mkdir(targetFolder)  
        if not os.path.exists(datasetFolder):
            os.mkdir(datasetFolder)   

        for i, f in enumerate(files):
            os.mkdir(f'./{targetFolder}/{i+1}')

    elif clf == 'multiClass':
        if not os.path.exists(targetFolder):
            os.mkdir(targetFolder)  
        
        for i, f in enumerate(files):
            os.mkdir(f'./{targetFolder}/{i+1}')


def get_and_save_roi(sourceFolder, targetFolder):
    files = os.listdir(sourceFolder)

    for i, f in enumerate(files):
        for img in os.listdir(f'./{sourceFolder}/{f}'):
            try:
                proie = PROIE()
                proie.extract_roi(f'./{sourceFolder}/{f}/{img}', rotate=True)
                proie.save(f'./{targetFolder}/{i+1}/{img}.jpg')
            except:
                continue
                

# def save_add_labels_and_side(folder):
#     files = os.listdir(folder)

#     data = []
#     labels = []
#     hand_sides = []

#     for f in files:
#         ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

#         for img, img_name in zip(ic, os.listdir(f'./{folder}/{f}')):

#             if img.shape[0] >= 128 and img.shape[1] >= 128:
#                 img = transform.resize(img, (128, 128))
#                 fd = hog(img, orientations=8, pixels_per_cell=(
#                     8, 8), cells_per_block=(1, 1))

#                 if 'r' in img_name:
#                     hand_side = 'r'
#                 else:
#                     hand_side = 'l'

#             data.append(fd)
#             labels.append(f)
#             hand_sides.append(hand_side)

#     with open('labels.txt', 'wb') as fp:
#         pickle.dump(labels, fp)
#     with open('hand_sides.txt', 'wb') as fp:
#         pickle.dump(hand_sides, fp)
#     with open('data.txt', 'wb') as fp:
#         pickle.dump(data, fp)


def save_add_labels(folder):
    files = os.listdir(folder)

    data = []
    labels = []

    for f in files:
        ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

        for img in ic:

            if img.shape[0] >= 128 and img.shape[1] >= 128:
                img = transform.resize(img, (128, 128))
                fd = hog(img, orientations=9, pixels_per_cell=(
                    8, 8), cells_per_block=(2, 2))

            data.append(fd)
            labels.append(f)

    with open(f'data_{folder}.txt', 'wb') as fp:
        pickle.dump(data, fp)
    with open(f'labels_{folder}.txt', 'wb') as fp:
        pickle.dump(labels, fp)


# def save_data_one_class(folder):
#     files = os.listdir(folder)

#     data = []
#     hand_sides = []

#     for f in files:
#         ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

#         for img, img_name in zip(ic, os.listdir(f'./{folder}/{f}')):

#             if img.shape[0] >= 128 and img.shape[1] >= 128:
#                 img = transform.resize(img, (128, 128))
#                 fd = hog(img, orientations=9, pixels_per_cell=(
#                     16, 16), cells_per_block=(3, 3))

#                 if 'r' in img_name:
#                     hand_side = 'r'
#                 else:
#                     hand_side = 'l'

#             data.append(fd)
#             hand_sides.append(hand_side)
            

#         with open(f'./oneClass_data/{f}.txt', 'wb') as fp:
#             pickle.dump(data, fp)

#         data = []
    
#     with open('hand_sides.txt', 'wb') as fp:
#         pickle.dump(hand_sides, fp)

def save_data_one_class_one_hand(sourceFolder, targetFolder):
    files = os.listdir(sourceFolder)

    data = []

    for f in files:
        ic = io.ImageCollection(f'./{sourceFolder}/{f}/*.jpg')

        for img, img_name in zip(ic, os.listdir(f'./{sourceFolder}/{f}')):

            if 'r' in img_name:
                if img.shape[0] >= 128 and img.shape[1] >= 128:
                    img = transform.resize(img, (128, 128))
                    fd = hog(img, orientations=9, pixels_per_cell=(
                        8, 8), cells_per_block=(2, 2))

                    data.append(fd)

        with open(f'./{targetFolder}/{f}.txt', 'wb') as fp:
            pickle.dump(data, fp)

        data = []


def import_data(directory):
    subjects = []

    for filename in os.listdir(directory):
        with open(f'{directory}/{filename}', 'rb') as fp:
            data = np.array(pickle.load(fp))
            subjects.append([data, filename])

    return subjects


def filter_and_pca_subjects(subjects, n_components):
    good_subjects = []  # n_components or more good ROIs
    pca = PCA(n_components=n_components)

    for subject in subjects:
        if subject[0].shape[0] >= n_components:
            data = subject[0]
            subject_number = subject[1].replace('.txt', '')

            pca.fit(data)
            data = pca.fit_transform(data)
            good_subjects.append([data, subject_number])

    return good_subjects


# save_add_labels('dataset')
# save_data_one_class('dataset')
# save_data_one_class_one_hand('dataset')
# create_dataset_folders('palms_data', 'oneClass_data')
