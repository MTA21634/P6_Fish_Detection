from tracker import *


# A multi-object tracker class
class multi_tracker:
    def __init__(self, a_T, o_T, v_T, h_T, c_T):
        self.trackers = []
        self.old_trackers = []
        self.id_count = 0
        self.a_T = a_T  # 100
        self.o_T = o_T  # 0
        self.v_T = v_T  # 20
        self.h_T = h_T
        self.c_T = c_T

    def update(self, contours, frame, img):
        for c in contours:
            initialized = False

            for t in reversed(self.trackers):
                if t.get_age(frame) < 6:
                    if t.check_if_similar(c, img):
                        t.update(c, frame)
                        initialized = True
                        break
                else:
                    self.old_trackers.append(t)
                    self.trackers.remove(t)

            if initialized is not True:
                new_tracker = tracker(c, frame, img, self.id_count, self.a_T, self.o_T, self.v_T, self.h_T, self.c_T)
                self.trackers.append(new_tracker)
                self.id_count += 1

    def end_tracking(self):
        for t in reversed(self.trackers):
            self.old_trackers.append(t)
            self.trackers.remove(t)

    def cull_trackers(self):
        for t in reversed(self.old_trackers):
            if t.check_if_noise(8):
                self.old_trackers.remove(t)
            else:
                continue

    def reorder_trackers(self):
        id_count = 0
        for t in self.old_trackers:
            t.id = id_count
            id_count += 1

    def get_contours_by_frame(self, frame):
        contours = []
        for t in self.trackers:
            contours.append(t.get_contour_by_frame(frame))
        return contours

    def get_old_trackers(self):
        return self.old_trackers

    def get_trackers(self):
        return self.trackers
