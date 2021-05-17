import random
from scipy.spatial import distance as dist
import numpy as np
import math
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


# Function for getting orientation of an object
def get_orientation(pts):
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    center = (int(mean[0, 0]), int(mean[0, 1]))
    angle = -int(np.rad2deg(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))) - 90  # orientation in Euler angles

    return angle


def create_roi(contour, img):
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w]
    return roi


def get_histogram(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


class tracker:
    def __init__(self, initial_object, initial_frame, initial_img, ID, a_T, o_T, v_T, h_T, c_T):
        self.positions = {initial_frame: initial_object}
        self.features_list = []
        self.id = ID
        self.global_a_T = a_T
        self.global_o_T = o_T
        self.global_v_T = v_T
        self.global_h_T = h_T
        self.global_c_T = c_T
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.add_new_feature(get_centroid(initial_object), cv2.contourArea(initial_object),
                             get_orientation(initial_object), initial_frame)
        self.hist = get_histogram(create_roi(initial_object, initial_img))

    def check_if_similar(self, contour, img):
        # Find the absolute difference between the sizes of contours
        area_diff = abs(self.features_list[-1]["area"] - cv2.contourArea(contour))

        # Find the velocity difference between two contours
        velocity_diff = magnitude(difference(get_centroid(contour), self.features_list[-1]["centroid"]))

        # Find the absolute rotation difference between two contours
        rotation_diff = abs(self.features_list[-1]["orientation"] - get_orientation(contour))

        # Check if the passed contour matches our tracked object
        if area_diff < self.global_a_T and velocity_diff < self.global_v_T and rotation_diff < self.global_o_T:
            # Calculate a new histogram for a passed contour
            new_hist = get_histogram(create_roi(contour, img))

            # Find the difference between two histograms based on Chebyshev's distance
            d = dist.chebyshev(self.hist, get_histogram(create_roi(contour, img)))

            # Find the similarity between two contours
            d2 = cv2.matchShapes(self.get_contour_by_frame(self.features_list[-1]["frame"]), contour, 1, 0.0)

            # Check if the overlap is significant
            if d < self.global_h_T and d2 < self.global_c_T:
                self.hist = new_hist
                return True
            else:
                return False
        else:
            return False

    def update(self, contour, frame):
        self.positions[frame] = contour
        self.add_new_feature(get_centroid(contour), cv2.contourArea(contour), get_orientation(contour), frame)

    def add_new_feature(self, c, a, o, f):
        feature_vector = {"centroid": c, "area": a, "orientation": o, "frame": f}
        self.features_list.append(feature_vector)

    def get_contour_by_frame(self, frame):
        try:
            return self.positions[frame]
        except KeyError:
            return None

    def check_if_noise(self, threshold):
        if len(self.positions) < threshold:
            return True
        else:
            return False

    def traveled_distance(self):
        total = 0
        previous_centroid = None

        for i, f in enumerate(self.features_list):
            if i > 0:
                distance = magnitude(addition(f["centroid"], previous_centroid))
                total += distance
            else:
                previous_centroid = f["centroid"]

        return total

    def get_age(self, frame):
        return frame - self.features_list[-1]["frame"]

    def get_id(self):
        return self.id
