# -*- coding: UTF-8 -*-

from PROIE import *
from functions import *
from skimage import io
from skimage.feature import hog
import os

if __name__ == '__main__':

    image = io.imread('./resources/palmprint_roi.jpg')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    plt.imshow(hog_image, cmap='gray')
    plt.savefig('./resources/hog1.jpg')

    def get_and_save_roi(folder_path):
        files = os.listdir(folder_path)

        for i, f in enumerate(files):
            for img in os.listdir(f'./{folder_path}/{f}'):
                try:
                    proie = PROIE()
                    proie.extract_roi(f'{folder_path}/{f}/{img}', rotate=True)
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

    def filter_bad_roi_get_hog(folder):
        img_collection  = io.ImageCollection(f'./{folder}/*/*.jpg')
        hogs = []
        
        for img in img_collection:
            if img.shape[0] < 100 or img.shape[1] < 100:
                continue
            else:
                fd = hog(img, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1))
                hogs.append(fd)
        
        return hogs

    # lbps = filter_bad_roi_get_lbp('dataset')
    # get_and_save_roi('./palms_data')