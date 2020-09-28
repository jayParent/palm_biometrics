import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

import skimage.data
import skimage.segmentation as seg
from skimage import filters, transform
from skimage import exposure, io, feature, measure
from skimage.feature import canny, local_binary_pattern, corner_harris, corner_peaks, corner_subpix
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte
from skimage.measure import find_contours
from scipy import ndimage as ndi


def importCollection(collection):
    img_list = io.ImageCollection('./palms_data/' + collection + '/*.jpg')
    return img_list


def find_corners(image, collection, i):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    corners = corner_peaks(corner_harris(
        image), min_distance=5, threshold_rel=0.015)

    ax1.imshow(image, cmap='gray')
    ax2.imshow(image, cmap='gray')

    ax2.plot(corners[:, 1], corners[:, 0], color='cyan', marker='o',
             linestyle='None', markersize=6)

    plt.savefig(f'./landmarks/test/{collection}/corner_detection{i}.jpg')

    return corners


def find_landmarks_v2(corners):
    corners_x_mean = np.mean(corners, axis=0)[1]
    corners_mean = np.mean(corners)
    print(f'corners x mean: {corners_x_mean} corners mean: {corners_mean}')
    potential_landmarks = []

    for corner in corners:
        corner_mean = np.mean(corner)
        print(f'corner: {corner} corner mean: {corner_mean} difference: {abs(corners_x_mean - corner[1])}')
        if abs(corners_x_mean - corner[1]) < 150:
            potential_landmarks.append(corner)
            
    potential_landmarks = np.array(potential_landmarks)
    print(potential_landmarks)
    print(np.amax(potential_landmarks, axis=0)[0])
    
    corners_y_mean = np.mean(potential_landmarks, axis=0)[0]

    landmark_candidates = []
    for point in potential_landmarks:
        if abs(corners_y_mean - point[0]) < 100:
            landmark_candidates.append(point)
    
    landmark_candidates = np.array(landmark_candidates)
    print(landmark_candidates)

    # for i, point in enumerate(potential_landmarks):
    #     distance_between_points = math.hypot(
    #         potential_landmarks[i + 1][1] - point[1], potential_landmarks[i + 1][0] - point[0])
    #     print(distance_between_points)


def get_roi_and_lbp_v2(image):
    pass


def find_edges_and_contours(image):
    mask = image > 50
    edges = feature.canny(mask)
    contours = measure.find_contours(edges, 0.8, fully_connected='high')

    return contours


def find_landmarks(contours):
    max_point = np.array([])
    max_points = []
    for contour in contours:
        if len(contour) < 700:
            max_mean = 0
            for p in contour:
                if p.mean() > max_mean:
                    max_point = p
            max_points.append(max_point)

    # Landmarks
    max_points = [max_points[0], max_points[-1]]
    landmark_1 = np.array(max_points[0])
    landmark_2 = np.array(max_points[1])

    return landmark_1, landmark_2


def get_roi_and_lbp(image):
    try:
        contours = find_edges_and_contours(image)

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

        contours = find_edges_and_contours(image)

        # palm not good, exit
        if len(contours) < 5:
            return

        landmark_1, landmark_2 = find_landmarks(contours)

        landmark_distance = math.hypot(
            landmark_2[1] - landmark_1[1], landmark_2[0] - landmark_1[0])

        # palm not good, exit
        if landmark_distance < 150 or landmark_distance > 220:
            return

        distance_from_fingers = landmark_distance / 5

        # Points slightly under landmark line, start of ROI
        top_left_point = np.array(
            [landmark_1[0] - distance_from_fingers, landmark_1[1]])
        top_right_point = np.array(
            [landmark_2[0] - distance_from_fingers, landmark_2[1]])

        if top_left_point[0] < landmark_1[0]:
            top_left_point = np.array(
                [landmark_1[0] + distance_from_fingers, landmark_1[1]])
            top_right_point = np.array(
                [landmark_2[0] + distance_from_fingers, landmark_2[1]])

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

        # Crop image to 128, 128. Region of interest
        miny = int(top_left_point[0])
        maxy = int(bottom_right_point[0])
        minx = int(bottom_left_point[1])
        maxx = int(top_right_point[1])

        roi = transform.resize(image[miny:maxy, minx:maxx], (128, 128))
        roi = img_as_ubyte(roi)

        # Local Binary Pattern
        radius = 1
        n_points = 8 * radius

        lbp = feature.local_binary_pattern(
            roi, n_points, radius, method='default')

        return lbp
    except:
        return
