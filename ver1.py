import os
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from scipy import ndimage as ndi
from skimage import feature, transform
from skimage import util, io, exposure, img_as_float, filters, segmentation, morphology, color, measure, img_as_ubyte
from skimage.measure import label
from functions import importCollection, findLandmarks, saveImage

# collection = '0005'
# img_collection = importCollection(collection)
# for i, img in enumerate(img_collection):
#     findLandmarks(img, i, collection)
    


palm_filename = os.path.join(skimage.data_dir, '0001_m_l_02.jpg')
palm = img_as_ubyte(io.imread(palm_filename, as_gray=True))

findLandmarks(palm, '3', 'test')



# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

# ax1.imshow(palm, cmap='gray')
# ax2.imshow(palm, cmap='gray')
# plt.savefig('img_ver1.jpg')
