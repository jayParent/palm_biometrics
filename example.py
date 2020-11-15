# -*- coding: UTF-8 -*-

from PROIE import *
from functions import *
from dataset_import import *
from skimage import io
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
import os
import pickle

if __name__ == '__main__':

    create_dataset_folders('palms_data')
    get_and_save_roi('palms_data')
    save_add_labels('palms_data')
