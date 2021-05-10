import math
import numpy as np
import cv2


def magnituted(vector_2d):
    (x, y) = vector_2d
    mag = math.sqrt(pow(x, 2) + pow(y, 2))
    return mag


def difference(a, b):
    diff = tuple(map(lambda i, j: i - j, a, b))
    return diff


def addition(a, b):
    add = tuple(map(lambda i, j: i + j, a, b))
    return add


def centroid(a):
    M = cv2.moments(a)
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center


def closest_pair(a, arr, max_dist):
    pair = None
    smallest_dist = max_dist
    center_a = centroid(a)

    for c in arr:
        center_c = centroid(c)
        diff = difference(center_c, center_a)
        dist = magnituted(diff)
        if dist < smallest_dist:
            smallest_dist = dist
            pair = c

    return pair


def detect_change(a, b, minimum_change):
    change = False

    center_a = centroid(a)
    center_b = centroid(b)

    diff = difference(center_a, center_b)
    dist = magnituted(diff)

    if dist > minimum_change:
        change = True

    return change


# def compare_contours(c1, c2):
#     same = True
#     c1 = np.array(c1)
#     c2 = np.array(c2)
#     try:
#         for i in range(c1.size):
#             if c1[i] != c2[i]:
#                 same = False
#     except IndexError:
#         return False
#
#     return same


# Multi object tracker based on euclidean distance
class euclidean_tracker:
    def __init__(self):
        self.previous_contours = []  # A list for storing previously passed contours
        self.objects = []  # An array that stores all tracked objects
        self.id_count = 0  # ID value for a new individual tracker

    def update(self, contours, frame):
        current_objects = []
        """----------------------------------------------------------------------------------------------------------"""
        # Loop through all of the detections that are passed
        for c in contours:
            initialized = False
            closest = closest_pair(c, self.previous_contours, 25)  # Find a closest pair within a set of contours in
            # a previous frame
            c_area = cv2.contourArea(c)
            for obj in self.objects:
                print(obj.get_contour_by_frame(frame - 1))
                if c_area == cv2.contourArea(obj.get_contour_by_frame(frame - 1)):
                    obj.update(c, frame, False)
                    initialized = True

            if initialized is False:
                new_object = fish_tracker(self.id_count)
                new_object.update(c, frame, False)
                current_objects.append(new_object)
                self.id_count += 1
        """----------------------------------------------------------------------------------------------------------"""
        for obj in self.objects:
            if obj.get_contour_by_frame(frame) is None:
                continue
            elif obj.get_consecutive_true_flags() > 10:
                continue
        """----------------------------------------------------------------------------------------------------------"""
        # Set previous contours to current contours
        self.previous_contours = contours

        # Append newly detected objects to a list of all objects
        for obj in current_objects:
            self.objects.append(obj)

        #
        # # New object initialized with current contours first value and current frame number
        # if closest is None:
        #     new_object = fish_tracker(self.id_count)
        #     new_object.add_new(c, frame)
        #     current_objects.append(new_object)
        #     self.id_count += 1
        # # Update the associated object's pool of contour locations
        # else:
        #     closest_area = cv2.contourArea(closest)
        #     for obj in self.objects:
        #         if frame - 1 in obj.positions:
        #             if closest_area == cv2.contourArea(obj.get_by_frame(frame - 1)):
        #                 obj.add_new(c, frame)

    def get_objects(self):
        return self.objects


class fish_tracker:
    def __init__(self, Id):
        self.positions = []  # Set of all contours that are associated to this object
        self.id = Id
        self.prediction_count = 0

    def update(self, contour, frame, state):
        self.positions.append((frame, contour, state))

    def get_contour_by_frame(self, frame):
        for p in self.positions:
            if p[0] == frame:
                return p[1]

        return None

    def get_consecutive_true_flags(self):
        flag_number = 0
        for p in reversed(self.positions):
            if p[2]:
                flag_number += 1
            else:
                return flag_number

        return flag_number

    def get_id(self):
        return self.id

    # def velocity_prediction(self, frame):
    #     if frame - 2 in self.positions:
    #         center_b = centroid(self.positions.get(frame - 1))
    #         center_a = centroid(self.positions.get(frame - 2))
    #         velocity = difference(center_b, center_a)
    #         prediction = addition(center_b, velocity)
    #         (x, y) = prediction
    #         predicted_contour = np.array([[x - 20, y - 20], [x + 20, y - 20], [x + 20, y + 20], [x - 20, y + 20]])
    #
    #         self.prediction_count += 1
    #         self.add_new(predicted_contour, frame)

# class EuclideanDistTracker:
#     def __init__(self):
#         # Store the center positions of the objects
#         self.center_points = {}
#         # Keep the count of the IDs
#         # each time a new object id detected, the count will increase by one
#         self.id_count = 0
#
#     def update(self, objects_rect):
#         # Objects boxes and ids
#         objects_bbs_ids = []
#
#         # Get center point of new object
#         for rect in objects_rect:
#             x, y, w, h = rect
#             cx = (x + x + w) // 2
#             cy = (y + y + h) // 2
#
#             # Find out if that object was detected already
#             same_object_detected = False
#             for id, pt in self.center_points.items():
#                 dist = math.hypot(cx - pt[0], cy - pt[1])
#
#                 if dist < 25:
#                     self.center_points[id] = (cx, cy)
#                     # print(self.center_points)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break
#
#             # New object is detected we assign the ID to that object
#             if same_object_detected is False:
#                 self.center_points[self.id_count] = (cx, cy)
#                 objects_bbs_ids.append([x, y, w, h, self.id_count])
#                 self.id_count += 1
#
#         # Clean the dictionary by center points to remove IDS not used anymore
#         new_center_points = {}
#         for obj_bb_id in objects_bbs_ids:
#             _, _, _, _, object_id = obj_bb_id
#             center = self.center_points[object_id]
#             new_center_points[object_id] = center
#
#         # Update dictionary with IDs not used removed
#         self.center_points = new_center_points.copy()
#         return objects_bbs_ids


# class fish_object:
#     def __init__(self, c, i):
#         self.positions = []
#         self.state = "fish"
#         self.id = i
#         self.add_position(c)
#
#     def add_position(self, contour):
#         self.positions.append(contour)
#
#     def detect_change(self, contours):
#
#         closest = closest_pair(self.positions[-1], contours, 100)
#
#         if closest is None:
#             self.positions.append(self.positions[-1])
#             # center_b = centroid(self.positions[-1])
#             # center_a = centroid(self.positions[-2])
#             # velocity = difference(center_b, center_a)
#             # prediction = addition(center_b, velocity)
#             # (x, y) = prediction
#             # predicted_contour = np.array([[x - 20, y - 20], [x + 20, y - 20], [x + 20, y + 20], [x - 20, y + 20]])
#             # self.positions.append(predicted_contour)
#         else:
#             self.positions.append(closest)
#
#     def traveled_path(self):
#         total_path = 0
#         previous_center = 0
#
#         for i in range(len(self.positions)):
#             center_i = centroid(self.positions[i])
#
#             if i > 0:
#                 diff = difference(center_i, previous_center)
#                 total_path += magnituted(diff)
#
#             previous_center = center_i
#
#         return total_path
#
#     def standard_deviation(self):
#         previous_center = 0
#         traveled_distances = []
#
#         for i in range(len(self.positions)):
#             center_i = centroid(self.positions[i])
#
#             if i > 0:
#                 diff = difference(center_i, previous_center)
#                 traveled_distances.append(magnituted(diff))
#
#             previous_center = center_i
#
#         mean = statistics.mean(traveled_distances)
#         sd = statistics.stdev(traveled_distances)
#         return sd, mean, traveled_distances
#
#     def asses_accuracy(self):
#         sd, mean, distances = self.standard_deviation()
#         accuracy = 1
#
#         for i in distances:
#             if i > mean + sd:
#                 accuracy -= 1 / len(distances)
#
#         return accuracy

# def make_prediction(self):


# def detect_fish(arr):
#     fishes = [fish_object(_, idx) for idx, _ in enumerate(arr[0])]
#     for contours in arr:
#         for x in reversed(range(len(fishes))):
#             fishes[x].detect_change(contours)
#             if fishes[x].state is None:
#                 fishes.pop(x)
#
#     return fishes
