from tracker import *
import math
import cv2

threshold = 0


def thresh_callback(val):
    global threshold
    threshold = val


def check_box_intersection(box1, box2):
    x1, y1, w1, h1 = cv2.boundingRect(box1)
    x2, y2, w2, h2 = cv2.boundingRect(box2)

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlapArea = x_overlap * y_overlap

    return overlapArea


def main():
    source_window = "Source"
    cv2.namedWindow(source_window)
    max_thresh = 255
    thresh = 80
    cv2.createTrackbar('Threshold:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)

    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=60)
    fgbg = cv2.createBackgroundSubtractorKNN(history=200)

    tracker = euclidean_tracker()

    cap = cv2.VideoCapture('test.mp4')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended, exiting program...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(gray)

        fgmask = fgbg.apply(hist)

        ret, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

        fgmask = cv2.medianBlur(thresh, 15)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_2)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_2)

        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in reversed(range(len(contours))):
            for j in range(i):
                intersection_area = check_box_intersection(contours[i], contours[j])
                print(intersection_area)
                smallest_box = min(cv2.contourArea(contours[i]), cv2.contourArea(contours[j]))
                if intersection_area >= smallest_box:
                    contours.pop(i)
                    break

        for c in contours:
            if cv2.contourArea(c) < 100000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(frame, str(cv2.contourArea(c)), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow(source_window, cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# circularity = cv2.arcLength(c, True) / (2 * math.sqrt(math.pi * cv2.contourArea(c)))
# x, y, w, h = cv2.boundingRect(c)
# compactness = cv2.contourArea(c) / (w * h)
#
# if circularity > 1.2 and compactness < 0.7:
#     x, y, w, h = cv2.boundingRect(c)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     fish_amount += 1
# else:
#     x, y, w, h = cv2.boundingRect(c)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

# print("Detected fish: " + str(fish_amount) + ", Frame: " + str(frame_number))
# print("\n")


# def main():
#     a = [4, 80, 12, 23, 5]
#     b = [41, 32, 2, 37, 8]
#
#     for c in b:
#         print(closest_pair(c, a, 50))


# if frame_number == 24:
#     closest = closest_pair(c, points, 40)
#     # circularity = cv2.arcLength(c, True) / (2 * math.sqrt(math.pi * cv2.contourArea(c)))
#     # area = cv2.contourArea(c)
#     # print(area)
#     x, y, w, h = cv2.boundingRect(c)
#     compactness = cv2.contourArea(c) / (w * h)
#     if closest is not None:
#         x, y, w, h = cv2.boundingRect(c)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

# if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 12 == 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) is not 0:
#     print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
#     detected_fish = detect_fish(points)
#     points = []
#     frame_number = 0
#     print("There were " + str(len(detected_fish)) + " detected, at " + str(
#         int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / 24) + " seconds")
#
# for fish in detected_fish:
#     distance = fish.traveled_path()
#     velocity = distance / 0.5
#     x, y, w, h = cv2.boundingRect(fish)
#     compactness = cv2.contourArea(fish) / (w * h)
#     if distance > 100 and fish.asses_accuracy() > 0.7:
#         x, y, w, h = cv2.boundingRect(fish.positions[frame_number])
#         cv2.putText(buffer[0], str(distance), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
#         cv2.rectangle(buffer[0], (x, y), (x + w, y + h), (0, 255, 0), 3)

# detections = []
# for c in contours:
#     # Calculate area and remove small elements
#     area = cv2.contourArea(c)
#     if area > 100:
#         # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
#         x, y, w, h = cv2.boundingRect(c)
#
#         detections.append([x, y, w, h])
#
# # 2. Object Tracking
# boxes_ids = tracker.update(detections)
# for box_id in boxes_ids:
#     x, y, w, h, Id = box_id
#     cv2.putText(frame, str(Id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
