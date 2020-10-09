# -*- coding: UTF-8 -*-

from PROIE import *
from functions import *
from skimage import io
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
import os
import pickle

if __name__ == '__main__':

    # image = io.imread('./resources/palmprint_roi.jpg')
    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
    #                 cells_per_block=(1, 1), visualize=True, multichannel=False)
    # plt.imshow(hog_image, cmap='gray')
    # plt.savefig('./resources/hog1.jpg')

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

    def filter_bad_roi_get_lbp(folder):
        img_collection  = io.ImageCollection(f'./{folder}/*/*.jpg')
        lbps = []
        
        for img in img_collection:
            if img.shape[0] < 100 or img.shape[1] < 100:
                continue
            else:
                lbp = get_lbp(img)
                lbps.append(lbp)
        
        return lbps
    
    def save_add_labels_and_side(folder):
        files = os.listdir(folder)

        labels = []
        hand_sides = []
        data = []

        for i, f in enumerate(files):
            ic = io.ImageCollection(f'./{folder}/{f}/*.jpg')

            for img_name in os.listdir(f'./{folder}/{f}'):
                if 'r' in img_name:
                    hand_side = 'r'
                else:
                    hand_side = 'l'

            for img in ic:
                if img.shape[0] >= 128 and img.shape[1] >= 128:
                    img = transform.resize(img, (128, 128))
                    fd = hog(img, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1))

                    data.append(fd)
                    labels.append(i+1)
                    hand_sides.append(hand_side)


        with open('labels.txt', 'wb') as fp:
            pickle.dump(labels, fp)
        with open('hand_sides.txt', 'wb') as fp:
            pickle.dump(hand_sides, fp)
        with open('data.txt', 'wb') as fp:
            pickle.dump(data, fp)

    def filter_bad_roi_get_hog(folder):
        img_collection  = io.ImageCollection(f'./{folder}/*/*.jpg')
        hogs = []
        
        for img in img_collection:
            if img.shape[0] < 128 or img.shape[1] < 128:
                continue
            else:
                img = transform.resize(img, (128, 128))
                fd = hog(img, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1))
                hogs.append(fd)
        
        return hogs

    def get_feature_descriptors(palms):
        fds = []

        for palm in palms:
            fd = hog(palm, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1))
            fds.append(fd)

        with open('hog_fds.txt', 'wb') as fp:
            pickle.dump(fds, fp)

    # with open('palms.txt', 'rb') as fp:
    #     palms = pickle.load(fp)
    # with open('labels.txt', 'rb') as fp:
    #     labels = pickle.load(fp)
    # with open('hand_sides.txt', 'rb') as fp:
    #     hand_sides = pickle.load(fp)
    # with open('hog_fds.txt', 'rb') as fp:
    #     hog_fds = pickle.load(fp)
    # with open('data.txt', 'rb') as fp:
    #     data = pickle.load(fp)