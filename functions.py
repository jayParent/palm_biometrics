import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

import skimage.data
import skimage.segmentation as seg
from skimage import filters, transform
from skimage import draw
from skimage import color
from skimage import exposure, io, feature, measure
from skimage.color import rgb2gray, label2rgb
from skimage.feature import hog, canny, local_binary_pattern, ORB, plot_matches, match_descriptors
from skimage.filters import sobel, prewitt, roberts
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, label, regionprops
from skimage.morphology import closing, square
from skimage.draw import rectangle, rectangle_perimeter
from scipy import ndimage as ndi

def importCollection(collection):
    img_list = io.ImageCollection('./palms_data/' + collection + '/*.jpg')
    return img_list
    # for i, img in enumerate(img_list):
    #     print('index: {}, shape: {}'.format(i, img.shape))
    #     cropped_img = cropImage(img, 5)
    #     saveImage(cropped_img, i, collection)

def cropImage(image, tolerance):
    y = int((image.shape[0]) / 2)
    x = int((image.shape[1]) / 2)

    seed_point = (y, x)
    flood_mask = seg.flood(image, seed_point, tolerance=tolerance)

    label_image = label(flood_mask)

    for region in regionprops(label_image, intensity_image=image):
        minr, minc, maxr, maxc = region.bbox
        cropped = image[minr:maxr, minc:maxc]
    
    return cropped

def cropUntilTargetMean(image):
    tolerance = 20
    palm_mean = 0
    target_mean = 132

    while palm_mean < target_mean:
        cropped_palm = cropImage(image, tolerance)
        palm_mean = cropped_palm.mean()
        print(palm_mean)
        tolerance -= 0.25
        plt.imshow(cropped_palm, cmap='gray')

    return cropped_palm

def maskAndEqualize(image):
    mask = image > 132
    masked_palm = image * mask
    palm_equalized = exposure.equalize_hist(masked_palm, nbins=256, mask=None)

    return palm_equalized

def saveImage(image, index, collection):
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.savefig('palms_cropped/' + collection + '/flood_fill' + str(index) + '.jpg')

def findLandmarks(image, index, collection):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))

    mask = image > 50
    edges = feature.canny(mask)

    contours = measure.find_contours(edges, 0.8, fully_connected='high')

    # Find longest contour
    max_length = 0
    for contour in contours:
        if len(contour) > max_length:
            max_length = len(contour)
            longest_contour = contour

    longest_contour_min_y = image.shape[0]
    longest_contour_max_y = 0

    # Rotate image if its horizontal
    for point in longest_contour:
        if point[0] < longest_contour_min_y:
            longest_contour_min_y = point[0]
        if point[0] > longest_contour_max_y:
            longest_contour_max_y = point[0]

    if longest_contour_max_y - longest_contour_min_y < image.shape[0]:
        image = img_as_ubyte(transform.rotate(image, -80))

    mask = image > 50
    edges = feature.canny(mask)

    contours = measure.find_contours(edges, 0.8, fully_connected='high')

    max_point = np.array([])
    max_points = []
    for i, contour in enumerate(contours):
        # print(f'contour: {i}', len(contour))
        if len(contour) < 700:
            max_mean = 0
            for p in contour:
                if p.mean() > max_mean:
                    max_point = p
            max_points.append(max_point)
        ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)

    # Landmarks
    max_points = [max_points[0], max_points[-1]]
    landmark_1 = np.array(max_points[0])
    landmark_2 = np.array(max_points[1])

    x_values = [landmark_1[1], landmark_2[1]]
    y_values = [landmark_1[0], landmark_2[0]]

    ax1.plot(x_values, y_values, linewidth=2)

    x1 = landmark_1[1]
    y1 = landmark_1[0]
    x2 = landmark_2[1]
    y2 = landmark_2[0]

    landmark_distance = math.hypot(x2 - x1, y2 - y1)
    slope = (y2 - y1) / (x2 - x1)
    # opp_reciprocal = (1 / slope) * -1
    distance_from_fingers = int(image.shape[0] / 8)


    # Points slightly under landmark line, start of ROI
    top_left_point = np.array([y1 - distance_from_fingers, x1])
    top_right_point = np.array([y2 - distance_from_fingers, x2])
    
    if top_left_point[0] < landmark_1[0]:
        top_left_point = np.array([y1 + distance_from_fingers, x1])
        top_right_point = np.array([y2 + distance_from_fingers, x2])

    x_values = [top_left_point[1], top_right_point[1]]
    y_values = [top_left_point[0], top_right_point[0]]

    ax2.plot(x_values, y_values, linewidth=2)

    # Bottom points, end of ROI 
    dx = top_left_point[1] - top_right_point[1]
    dy = top_left_point[0] - top_right_point[0]
    dx /= landmark_distance
    dy /= landmark_distance
    x3 = top_left_point[1] + (landmark_distance) * dy
    y3 = top_left_point[0] - (landmark_distance) * dx
    x4 = top_right_point[1] + (landmark_distance) * dy
    y4 = top_right_point[0] - (landmark_distance) * dx
    
    bottom_left_point = np.array([y3, x3])
    bottom_right_point = np.array([y4, x4])

    x_values = [bottom_left_point[1], bottom_right_point[1]]
    y_values = [bottom_left_point[0], bottom_right_point[0]]

    ax2.plot(x_values, y_values, linewidth=2)

    # Crop image to 128, 128. Region of interest
    miny = int(top_left_point[0])
    maxy = int(bottom_right_point[0])
    minx = int(bottom_left_point[1])
    maxx = int(top_right_point[1])
    print(top_left_point)

    roi = transform.resize(image[miny:maxy, minx:maxx], (128, 128))
    roi = img_as_ubyte(roi)

    # Tests
    # p2, p98 = np.percentile(roi, (2, 98))
    # roi_rescaled = exposure.rescale_intensity(roi, in_range=(p2, p98))

    # roi_edges = feature.canny(roi_rescaled, sigma=1, low_threshold=50, high_threshold=150)
    # img_adapteq = img_as_ubyte(exposure.equalize_adapthist(roi, clip_limit=0.015))

    # fd, hog_image = hog(roi, orientations=8, pixels_per_cell=(8, 8),
    #                 cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    # Local Binary Pattern
    radius = 1
    n_points = 8 * radius

    lbp = feature.local_binary_pattern(roi, n_points, radius, method='default')
    
    ax1.imshow(image, cmap='gray')
    ax2.imshow(image, cmap='gray')
    ax3.imshow(roi, cmap='gray')
    ax4.imshow(lbp, cmap='gray')
    # ax4.hist(roi.ravel(), bins=256, histtype='step', color='black')
    ax5.hist(lbp.ravel(), bins='auto', histtype='step', color='black')
    # ax5.imshow(roi_rescaled, cmap='gray')
    # ax6.imshow(lbp, cmap='gray')
    ax6.hist(lbp.ravel(), bins=256, histtype='step', color='black')
    plt.savefig('landmarks/' + collection + '/' + str(index) + '.jpg')