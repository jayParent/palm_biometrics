import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
from numpy import load
from skimage import io
from sklearn.model_selection import train_test_split

collection = os.listdir('./dataset')

data = []
labels = []

# for folder in collection:
#     filenames = os.listdir(f'./dataset/{folder}')
#     for fname in filenames:
#         try:
#             lbp = np.load(f'./dataset/{folder}/{fname}', allow_pickle=True)
#             if lbp.shape == (128, 128):
#                 data.append(lbp)
#                 labels.append(folder)
#             else:
#                 continue
#         except:
#             continue

# with open('data.txt', 'wb') as fp:
#     pickle.dump(data, fp)
# with open('labels.txt', 'wb') as fp:
#     pickle.dump(labels, fp)

with open('data.txt', 'rb') as fp:
    data = pickle.load(fp)
with open('labels.txt', 'rb') as fp:
    labels = pickle.load(fp)

lbps_train, lbps_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1, random_state=42)