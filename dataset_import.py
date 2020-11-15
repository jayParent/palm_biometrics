# -*- coding: UTF-8 -*-

from PROIE import *
from functions import *
from skimage import io
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
import os
import pickle


def create_dataset_folders(folder):
    files = os.listdir(folder)

    os.mkdir('dataset')
    for i, f in enumerate(files):
        os.mkdir(f'./dataset/{i+1}')


def get_and_save_roi(folder):
    files = os.listdir(folder)

    for i, f in enumerate(files):
        for img in os.listdir(f'./{folder}/{f}'):
            try:
                proie = PROIE()
                proie.extract_roi(f'{folder}/{f}/{img}', rotate=True)
                proie.save(f'./dataset/{i+1}/{img}.jpg')
            except:
                continue


def save_add_labels_and_side(folder):
    files = os.listdir(folder)

    data = []
    labels = []
    hand_sides = []

    for f in files:
        ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

        for img, img_name in zip(ic, os.listdir(f'./{folder}/{f}')):

            if img.shape[0] >= 128 and img.shape[1] >= 128:
                img = transform.resize(img, (128, 128))
                fd = hog(img, orientations=8, pixels_per_cell=(
                    8, 8), cells_per_block=(1, 1))

                if 'r' in img_name:
                    hand_side = 'r'
                else:
                    hand_side = 'l'

            data.append(fd)
            labels.append(f)
            hand_sides.append(hand_side)

    with open('labels.txt', 'wb') as fp:
        pickle.dump(labels, fp)
    with open('hand_sides.txt', 'wb') as fp:
        pickle.dump(hand_sides, fp)
    with open('data.txt', 'wb') as fp:
        pickle.dump(data, fp)


def save_add_labels(folder):
    files = os.listdir(folder)

    data = []
    labels = []

    for f in files:
        ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

        for img in ic:

            if img.shape[0] >= 128 and img.shape[1] >= 128:
                img = transform.resize(img, (128, 128))
                fd = hog(img, orientations=8, pixels_per_cell=(
                    8, 8), cells_per_block=(1, 1))

            data.append(fd)
            labels.append(f)

    with open(f'data_{folder}.txt', 'wb') as fp:
        pickle.dump(data, fp)
    with open(f'labels_{folder}.txt', 'wb') as fp:
        pickle.dump(labels, fp)
