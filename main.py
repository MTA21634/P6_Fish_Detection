from numba import njit
from multi_tracker import *
import numpy as np
import cv2


@njit
def subtract_absolute(img1, img2):
    (height, width) = img1.shape
    result = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            diff = max(img1[i, j], img2[i, j]) - min(img1[i, j], img2[i, j])

            result[i, j] = diff

    return result


# Function for finding an overlapping area between two rectangles
def check_box_intersection(box1, box2):
    x1, y1, w1, h1 = cv2.boundingRect(box1)
    x2, y2, w2, h2 = cv2.boundingRect(box2)

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlapArea = x_overlap * y_overlap

    return overlapArea


def main():
    # Kernel for performing morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))

    # Initializing the background subtractor
    subtract = cv2.createBackgroundSubtractorMOG2(history=200)

    # Opening the video file
    cap = cv2.VideoCapture('test.mp4')

    # File for storing the recorded video
    result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (640, 360))

    """Initializing a multi object tracker. The arguments for the tracker are: Maximum allowed area difference,
    rotation difference, velocity difference, histogram difference and contour difference, respectively"""
    track = multi_tracker(25000, 180, 80, 0.45, 0.35)

    # Reading first frame and applying histogram equalization to it
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)

    # Converting the image to a 32-bit floating point. For storing the running average of all the frames
    avg = np.float32(hist)

    # A boolean for checking if the video is being analyzed
    analyzing = True

    # A variable for storing the the frame rate
    frame_rate = 1

    # A variable for storing all the final trackers
    old_trackers = None

    print("Analyzing video...")

    # While loop that analyzes given video, initializes trackers and after displays the video
    while True:
        ret, frame = cap.read()

        # If it is the last frame of the video then we end tracking and further analyze trackers
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 and analyzing:
            # End tracking
            track.end_tracking()

            # Remove trackers that have been tracking noise
            track.cull_trackers()

            # Reorder the id's of the trackers
            track.reorder_trackers()

            # Get the good trackers
            old_trackers = track.get_old_trackers()

            # Set the state to display
            analyzing = False

            # Play the video again by setting the current frame to 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Set the frame rate for displaying the video
            frame_rate = 44
            print("Analysis done, showing video.\nNumber of fish detected in the footage: " + str(len(old_trackers)) + " fish.")

        # If there are no frames to display, that means that video has ended
        if not ret:
            print("Ending video...")
            break

        if analyzing:
            # Converting the image to grayscale and equalizing the image histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.equalizeHist(gray)

            # Updating the running average
            cv2.accumulateWeighted(hist, avg, 0.08)

            # Converting running average from 32-bit floating point to 8-bit int
            res_avg = cv2.convertScaleAbs(avg)

            # Getting a second foreground mask
            diff = subtract_absolute(res_avg, hist)
            __, fg_mask1 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Extracting the foreground objects
            fg_mask2 = subtract.apply(hist)

            # Creating a second foreground mask
            __, fg_mask2 = cv2.threshold(fg_mask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Combining both masks into one
            combined_fg_mask_og = cv2.bitwise_and(fg_mask1, fg_mask2)

            # Erosion to remove noise
            combined_fg_mask = cv2.erode(combined_fg_mask_og, kernel, iterations=4)

            # Closing to bring objects back to their original size
            combined_fg_mask = cv2.morphologyEx(combined_fg_mask, cv2.MORPH_CLOSE, kernel, iterations=4)

            # Dilate objects so that we can use opening on them
            combined_fg_mask = cv2.dilate(combined_fg_mask, kernel, iterations=9)

            # Opening to further remove smaller objects
            combined_fg_mask = cv2.morphologyEx(combined_fg_mask, cv2.MORPH_OPEN, kernel, iterations=9)

            # Median blurring to smooth out the blocky objects
            combined_fg_mask = cv2.medianBlur(combined_fg_mask, 17)

            # Extracting objects from the frame
            contours, _ = cv2.findContours(combined_fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            track.update(contours, int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame)

            trackers = track.get_trackers()

            for t in trackers:
                c = t.get_contour_by_frame(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if c is not None:
                    if len(t.features_list) > 1:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.putText(frame, "#ID: " + str(t.get_id()), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], 0, t.color, 2)
                    else:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.putText(frame, "#ID: " + str(t.get_id()), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], 0, t.color, 2)

            cv2.imshow("output", cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))

        else:
            for t in old_trackers:
                c = t.get_contour_by_frame(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                if c is not None:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.putText(frame, "#ID: " + str(t.get_id()), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, t.color, 2)

            result.write(frame)
            cv2.imshow("output", cv2.resize(frame, (960, 544), interpolation=cv2.INTER_AREA))

        if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()