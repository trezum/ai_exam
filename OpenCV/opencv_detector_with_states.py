# importing OpenCV, time and Pandas library
import cv2
import numpy
import time

# importing haarcascades for face, eye and smile detction
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Colors are B G R
lower_blue = numpy.array([78, 158, 124])
upper_blue = numpy.array([138, 255, 255])
lower_red = numpy.array([0, 50, 50])
upper_red = numpy.array([10, 255, 255])

# -1 = quit
#  0 = motion detection
#  1 = pedestrian detection
#  2 = face detection
state = 0

# Capturing video
video = cv2.VideoCapture(0)


def sortbyy(e):
    return e[1]


def select_eyes(alleyes):
    # Keeps just two eyes with similar y values
    # TODO eyes should be in the top half of face.
    if len(alleyes) <= 2:
        return alleyes

    sortedeyes = sorted(alleyes, key=sortbyy)
    smallestdif = 1000000  # should be image max size
    selectedindex = 0
    i = 0
    while i < len(sortedeyes)-1:
        if sortedeyes[i+1][1] - sortedeyes[i][1] < smallestdif:
            smallestdif = sortedeyes[i+1][1] - sortedeyes[i][1]
            selectedindex = i
        i += 1
    return [sortedeyes[selectedindex], sortedeyes[selectedindex+1]]


def select_smile(smiles, selectedeyes):
    # Keeps just one smile with proper distance from eyes
    # https://upload.wikimedia.org/wikipedia/commons/0/06/AvgHeadSizes.png
    # Used to calculate distance between eyes and smile from 5th and 95th percentile.
    # TODO Eyes and smiles should not overlap or be contained within eachother
    if len(smiles) <= 1:
        return smiles
    if len(selectedeyes) != 2:
        return []

    # calculates the distance between the eyes outer edge horrisontaly
    if selectedeyes[0][0] > selectedeyes[1][0]:
        distancebetweeneyes = selectedeyes[0][0] + selectedeyes[0][2] - selectedeyes[1][0]
    else:
        distancebetweeneyes = selectedeyes[1][0] + selectedeyes[1][2] - selectedeyes[0][0]

    distancefromtoptoeyes = (selectedeyes[0][1] + selectedeyes[1][1]) / 2
    smilerangelow = distancebetweeneyes + distancefromtoptoeyes * 1.054
    smilerangehigh = distancebetweeneyes + distancefromtoptoeyes * 1.244

    # Create a new array with smiles fitting the average human between the 5th and 95th percentile
    verticalyselectedsmiles = []

    for (ex2, ey2, ew2, eh2) in smiles:
        if (ey2 + eh2) + distancefromtoptoeyes >= smilerangelow + distancefromtoptoeyes:
            if (ey2 + eh2) + distancefromtoptoeyes <= smilerangehigh + distancefromtoptoeyes:
                verticalyselectedsmiles.append([ex2, ey2, ew2, eh2])

    if len(verticalyselectedsmiles) <= 1:
        return verticalyselectedsmiles

    centerofeyes = (selectedeyes[0][0] + selectedeyes[1][0]) / 2
    distance = 1000000  # should be image max size
    smileindex = 0
    i = 0
    while i < len(verticalyselectedsmiles):
        horizontalcenterofsmile = (ew/2) + ex
        distancefromcenterofeyestosmilecenter = horizontalcenterofsmile - centerofeyes
        if distancefromcenterofeyestosmilecenter > 0:
            if distancefromcenterofeyestosmilecenter < distance:
                smileindex = i
                distance = distancefromcenterofeyestosmilecenter
        i += 1

    return [verticalyselectedsmiles[smileindex]]


def face_eyes_smile_detection():
    global ew
    global ex
    while True:
        _, frame = video.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            gray_face = gray_frame[y:y + h, x:x + w]  # cut the gray face frame out
            face = frame[y:y + h, x:x + w]  # cut the face frame out
            eyes = eye_cascade.detectMultiScale(gray_face)
            selectedeyes = select_eyes(eyes)
            for (ex, ey, ew, eh) in selectedeyes:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 2)

            smiles = smile_cascade.detectMultiScale(gray_face)
            for (ex, ey, ew, eh) in select_smile(smiles, selectedeyes):
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (225, 0, 255), 2)

        cv2.putText(frame, 'Face, eyes & smile', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow("Detector with states", frame)

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            return -1
        # if n entered change state to motion detection
        if key == ord('n'):
            return 0


def find_biggest(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Removing noise. Isolation of individual elements and joining disparate elements in an image.
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours = cv2.findContours(mask .copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) != 0:
        return max(contours, key=cv2.contourArea)
    return contours


def find_color(frame):
    biggest_orange = find_biggest(frame, lower_red, upper_red)
    biggest_blue = find_biggest(frame, lower_blue, upper_blue)

    if len(biggest_orange) != 0 and len(biggest_blue) != 0:
        if cv2.contourArea(biggest_blue) > cv2.contourArea(biggest_orange):
            return 110, 0, 0
        else:
            return 87, 109, 251
    elif len(biggest_orange) != 0:
        return 87, 109, 251
    elif len(biggest_blue) != 0:
        return 110, 0, 0


def pedestrian_detection():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    while True:
        active_color = (0, 255, 0)
        r, frame = video.read()
        if r:
            frame = cv2.resize(frame, (640, 360))  # Downscale to improve frame rate
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

            rects, weights = hog.detectMultiScale(gray_frame)

            for i, (x, y, w, h) in enumerate(rects):
                if weights[i] < 0.7:
                    continue
                pedestrian_frame = frame[y:y + h, x:x + w]
                active_color = find_color(pedestrian_frame)
                cv2.rectangle(frame, (x, y), (x + w, y + h), active_color, 2)

            cv2.putText(frame, 'Pedestrian', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)
            cv2.imshow("Detector with states", frame)

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            return -1
        # if n entered change state to face detection
        if key == ord('n'):
            return 2


def motion_detection():
    start = time.time()
    # Assigning our static_back to None
    static_back = None
    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

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

        if (abs(start - time.time()) > 2):
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

        # Use for drawing the found conturs
        # cv2.drawContours(frame,cnts,-1,(0, 0, 255))

        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)



        # Displaying the frame
        cv2.putText(frame, 'Motion', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Detector with states", frame)

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            return -1
        # if n entered change state to pedestrian detection
        if key == ord('n'):
            return 1


while True:
    if state == -1:
        break
    if state == 0:
        state = motion_detection()
    if state == 1:
        state = pedestrian_detection()
    if state == 2:
        state = face_eyes_smile_detection()

video.release()

# Destroying all the windows
cv2.destroyAllWindows()
