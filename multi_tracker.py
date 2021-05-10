from tracker import *
import math
import numpy as np
import cv2


# Function that finds a magnitude of a 2d vector
def magnitude(vector_2d):
    (x, y) = vector_2d
    mag = math.sqrt(pow(x, 2) + pow(y, 2))
    return mag


# Function that calculates the difference between two 2d vectors
def difference(a, b):
    diff = tuple(map(lambda i, j: i - j, a, b))
    return diff


# Function that sums up two 2d vectors
def addition(a, b):
    add = tuple(map(lambda i, j: i + j, a, b))
    return add


# Function that calculates a center of a contour
def get_centroid(c):
    M = cv2.moments(c)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center


# Function that finds closest object for a given object based on Euclidean distance
def closest_pair(a, arr, max_dist):
    pair = None
    smallest_dist = max_dist
    center_a = get_centroid(a)

    for c in arr:
        center_c = get_centroid(c)
        diff = difference(center_c, center_a)
        dist = magnitude(diff)
        if dist < smallest_dist:
            smallest_dist = dist
            pair = c

    return pair


# A multi-object tracker class
class offline_multi_tracker:
    def __init__(self, objects):
        self.objects = objects
        self.trackers = []
        self.id_count = 0

# # Multi object tracker based on euclidean distance
# class euclidean_tracker:
#     def __init__(self):
#         self.previous_contours = []  # A list for storing previously passed contours
#         self.objects = []  # An array that stores all tracked objects
#         self.id_count = 0  # ID value for a new individual tracker
#
#     def update(self, contours, frame):
#         current_objects = []
#         """----------------------------------------------------------------------------------------------------------"""
#         # Loop through all of the detections that are passed
#         for c in contours:
#             initialized = False
#             closest = closest_pair(c, self.previous_contours, 25)  # Find a closest pair within a set of contours in
#             # a previous frame
#             c_area = cv2.contourArea(c)
#             for obj in self.objects:
#                 print(obj.get_contour_by_frame(frame - 1))
#                 if c_area == cv2.contourArea(obj.get_contour_by_frame(frame - 1)):
#                     obj.update(c, frame, False)
#                     initialized = True
#
#             if initialized is False:
#                 new_object = fish_tracker(self.id_count)
#                 new_object.update(c, frame, False)
#                 current_objects.append(new_object)
#                 self.id_count += 1
#         """----------------------------------------------------------------------------------------------------------"""
#         for obj in self.objects:
#             if obj.get_contour_by_frame(frame) is None:
#                 continue
#             elif obj.get_consecutive_true_flags() > 10:
#                 continue
#         """----------------------------------------------------------------------------------------------------------"""
#         # Set previous contours to current contours
#         self.previous_contours = contours
#
#         # Append newly detected objects to a list of all objects
#         for obj in current_objects:
#             self.objects.append(obj)
#
#         #
#         # # New object initialized with current contours first value and current frame number
#         # if closest is None:
#         #     new_object = fish_tracker(self.id_count)
#         #     new_object.add_new(c, frame)
#         #     current_objects.append(new_object)
#         #     self.id_count += 1
#         # # Update the associated object's pool of contour locations
#         # else:
#         #     closest_area = cv2.contourArea(closest)
#         #     for obj in self.objects:
#         #         if frame - 1 in obj.positions:
#         #             if closest_area == cv2.contourArea(obj.get_by_frame(frame - 1)):
#         #                 obj.add_new(c, frame)
#
#     def get_objects(self):
#         return self.objects

# def detect_change(a, b, minimum_change):
#     change = False
#
#     center_a = get_centroid(a)
#     center_b = get_centroid(b)
#
#     diff = difference(center_a, center_b)
#     dist = magnitude(diff)
#
#     if dist > minimum_change:
#         change = True
#
#     return change
