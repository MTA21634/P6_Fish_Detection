from multi_tracker import *
import cv2


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
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # fg_mask = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=60)

    # Initializing background subtraction
    subtract = cv2.createBackgroundSubtractorKNN(history=200)

    # tracker = euclidean_tracker()

    # A dictionary for storing all the extracted objects from the footage
    all_contours = {}

    # Opening the video file
    cap = cv2.VideoCapture('test.mp4')

    # While loop that reads and displays each frame of the video
    while True:
        ret, frame = cap.read()

        # If there are no frames to display, that means that video has ended
        if not ret:
            print("Video ended, exiting program...")
            break

        # Converting the image to grayscale and equalizing the image histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(gray)

        # Extracting the foreground objects
        fg_mask = subtract.apply(hist)

        # Threshold to remove detected shadows
        ret, thresh = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)

        # Median blurring to remove noise cause by the background subtraction
        fg_mask = cv2.medianBlur(thresh, 15)

        # Morphological closing to close the holes in the objects
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_2)

        # Morphological opening to further remove noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_2)

        # Extracting objects from the frame
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Checking for overlapping contours and removing the smaller contour
        for i in reversed(range(len(contours))):
            for j in range(len(contours) - (len(contours) - i)):
                intersection_area = check_box_intersection(contours[i], contours[j])
                smallest_box = min(cv2.contourArea(contours[i]), cv2.contourArea(contours[j]))
                overlap_ratio = (100 * intersection_area) / smallest_box
                if overlap_ratio >= 50:
                    contours.pop(i)
                    break

        for c in contours:
            if cv2.contourArea(c) < 100000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(frame, str(cv2.contourArea(c)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("output", cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))

        all_contours[int(cap.get(cv2.CAP_PROP_POS_FRAMES))] = contours

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
