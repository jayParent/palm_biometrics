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


def find_corners(image):
    corners = corner_peaks(corner_harris(
        image), min_distance=5, threshold_rel=0.015)

    return corners


def find_landmarks_v2(corners):
    distance_between_points = []

    for i, corner in enumerate(corners):
        for j in range(len(corners)):
            distance = math.hypot(
                corners[j][1] - corner[1], corners[j][0] - corner[0])
            distance_between_points.append([i, j, distance])

    landmarks = []
    for dbp in distance_between_points:
        distance = dbp[2]
        corner1 = corners[dbp[0]]
        corner2 = corners[dbp[1]]
        if 170 < distance < 200:
            if abs(corner1[1] - corner2[1]) < 40:
                landmarks.append([corner2, corner1])
                break

    landmarks = np.array(landmarks)
    print(landmarks)
    return landmarks


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


def get_lbp(roi):
    radius = 1
    n_points = 8 * radius

    lbp = feature.local_binary_pattern(
        roi, n_points, radius, method='default')
        
    return lbp


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


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

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())