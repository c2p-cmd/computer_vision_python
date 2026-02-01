import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(
    "../../resources/DATA/haarcascades/haarcascade_frontalface_default.xml"
)


def detect_image(img):
    face_img = img.copy()
    face_rects = face_classifier.detectMultiScale(face_img)

    for x, y, w, h in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 7)

    return face_img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)

    frame = detect_image(frame)

    cv2.imshow("Live Face Detection", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
