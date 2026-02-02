import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open capture")
    exit(-1)


face_cascade = cv2.CascadeClassifier(
    "../../resources/DATA/haarcascades/haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape

    face_rects = face_cascade.detectMultiScale(frame, minSize=(45, 45))

    if len(face_rects) > 0:
        cv2.putText(
            frame,
            "Press C to continue",
            (int(width / 3), height - 20),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (100, 255, 100),
            5,
            cv2.LINE_AA,
        )

    cv2.imshow("First Capture", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("c"):
        break
    if k == ord("q") or k == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

cv2.destroyWindow("First Capture")

(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)

roi = frame[face_y : face_y + h, face_x : face_x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if ret:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)

        # meanshit or camshift
        ret, track_window = cv2.CamShift(dst, track_window, criteria)

        pts = cv2.boxPoints(ret).astype(int)
        img2 = cv2.polylines(frame, [pts], True, (0, 100, 255), 5)
        #

        cv2.imshow("Image", img2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break
    else:
        print("Broken!")
        break

cap.release()
cv2.destroyAllWindows()
