import os
import skimage
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from skimage import io
from sklearn.model_selection import train_test_split

collection = os.listdir('./dataset')

lbps = []
labels = []

for folder in collection:
    files = os.listdir(f'./dataset/{folder}')
    for f in files:
        try:
            data = load(f'./dataset/{folder}/{f}')
            lbps.append(data)
            labels.append(folder)
        except:
            continue

lbps_train, lbps_test, labels_train, labels_test = train_test_split(lbps, labels, test_size=0.1, random_state=42)
