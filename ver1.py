import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import save
from functions import importCollection, find_landmarks, get_roi_and_lbp, find_corners, find_landmarks_v2

collection = '0001'
img_collection = importCollection(collection)

for i, img in enumerate(img_collection):
    try:
        if i == 5:
            corners = find_corners(img, collection, i)
            find_landmarks_v2(corners)
    except:
        continue

# for i, img in enumerate(img_collection):
#     try:
#         lbp = get_roi_and_lbp(img)
#         plt.imshow(lbp, cmap='gray')
#         plt.savefig(f'./landmarks/{collection}/{i}.jpg')
#     except:
#         continue

# os.makedirs(f'./dataset/{i+1}')

# files = os.listdir('./palms_data')
# img_collections = []

# for fi in files:
#     img_collection = io.ImageCollection(f'./palms_data/{fi}/*.jpg')
#     img_collections.append(img_collection)

# for i, collection in enumerate(img_collections):
#     for j, img in enumerate(collection):
#         try:
#             lbp = getRoiAndLbp(img)
#             save(f'./dataset/{i+1}/{j}.npy', lbp, allow_pickle=True)
#         except:
#             continue
        