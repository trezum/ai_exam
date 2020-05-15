# importing OpenCV, time and Pandas library
import cv2
import numpy
import time

# Assigning our static_back to None
static_back = None

# List when any moving object appear
motion_list = [None, None]
start = time.time()

# Capturing video
video = cv2.VideoCapture(0)

# Infinite while loop to treat stack of image as video
while True:
    # Reading frame(image) from video
    check, frame = video.read()

    # Initializing motion = 0(no motion)
    motion = 0

    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In first iteration we assign the value
    # of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue

    if (abs( start - time.time())>2):
        static_back = gray
        start = time.time()

    # Difference between static background
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Finding contour of moving object
    (image, cnts, hier) = cv2.findContours(thresh_frame.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    # Displaying the frame
    cv2.imshow("Detector with states", frame)

    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        break
    if key == ord('n'):
        break
video.release()

# Destroying all the windows
cv2.destroyAllWindows()