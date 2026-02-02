import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise PermissionError("Capture not running!")

ret, frame1 = cap.read()
prev_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[:, :, 1] = 255

while True:
    ret, frame2 = cap.read()

    next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_img,
        next=next_img,
        flow=np.zeros((1, 1)),
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    magnitude, angle = cv2.cartToPolar(
        flow[:, :, 0],
        flow[:, :, 1],
        angleInDegrees=True,
    )

    hsv_mask[:, :, 0] = angle / 2
    hsv_mask[:, :, 2] = cv2.normalize(
        magnitude,
        np.zeros((1, 1)),
        0,
        255,
        cv2.NORM_MINMAX,
    )

    bgr_frame = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("Optical Flow", bgr_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):
        break

    prev_img = next_img

cap.release()
cv2.destroyAllWindows()
