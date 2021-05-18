"""
This file contains a class for multiple object tracking. The class associates detection that are passed to it,
based on the parameters this class has been initialized. It does so by instantiating 'tracker' objects, and passing
them their associated detections.

This class has been made by MTA21634, for a sixth semester project in Medialogy, AAU.
"""

from tracker import *


# A multi-object tracker class
class multi_tracker:
    def __init__(self, a_T, o_T, v_T, h_T, c_T):
        self.trackers = []
        self.old_trackers = []
        self.id_count = 0
        self.a_T = a_T  # Area difference threshold
        self.o_T = o_T  # Rotation difference threshold
        self.v_T = v_T  # Velocity difference threshold
        self.h_T = h_T  # Histogram difference threshold
        self.c_T = c_T  # Contour difference threshold

    # Function for updating the multi-object tracker with new detections
    def update(self, contours, frame, img):
        for c in contours:
            initialized = False

            for t in reversed(self.trackers):
                # Check if tracker had any recent observations in the past 0.5 s
                if t.get_age(frame) < 12:

                    # Pass the current contour to each tracker and check if the detection is similar to tracked object
                    if t.check_if_similar(c, img):
                        # If so then update the corresponding tracker with this detection
                        t.update(c, frame)
                        initialized = True
                        break
                # If not, then discard the tracker and add it to a list of old trackers, for later analysis
                else:
                    self.old_trackers.append(t)
                    self.trackers.remove(t)

            # Otherwise, create a new tracker for this detection
            if initialized is not True:
                new_tracker = tracker(c, frame, img, self.id_count, self.a_T, self.o_T, self.v_T, self.h_T, self.c_T)
                self.trackers.append(new_tracker)
                self.id_count += 1

    # Sets all the trackers to be old, and adds them to the old tracker list
    def end_tracking(self):
        for t in reversed(self.trackers):
            self.old_trackers.append(t)
            self.trackers.remove(t)

    # Remove trackers that might be faulty or might be tracking noise
    def cull_trackers(self):
        for t in reversed(self.old_trackers):
            if t.check_if_noise(12):
                self.old_trackers.remove(t)
            else:
                continue

    # Reorder the old tracker id's for convenience
    def reorder_trackers(self):
        id_count = 0
        for t in self.old_trackers:
            t.id = id_count
            id_count += 1

    # Get all the contours that appeared at specified frame
    def get_contours_by_frame(self, frame):
        contours = []
        for t in self.trackers:
            contours.append(t.get_contour_by_frame(frame))
        return contours

    # Get old trackers
    def get_old_trackers(self):
        return self.old_trackers

    # Get current trackers
    def get_trackers(self):
        return self.trackers
