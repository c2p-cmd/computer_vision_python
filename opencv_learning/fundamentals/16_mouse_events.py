import cv2
import numpy as np
import time

WINDOW_NAME = "My Drawing"
_time = 0


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(
            img=img,
            center=(x, y),
            radius=100,
            color=(0, 255, 100),
            thickness=-1,  # filled circle
        )


def draw_circle_adv(event, x, y, flags, param):
    global _time
    radius = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        _time = time.time_ns()
    if event == cv2.EVENT_LBUTTONUP:
        elapsed = time.time_ns() - _time
        radius = int(elapsed // 10_000_000)
        print("Radius ", radius)
        cv2.circle(
            img=param,
            center=(x, y),
            radius=radius,
            color=(0, 255, 100),
            thickness=-1,  # filled circle
        )


img = np.zeros((512, 512, 3))

cv2.namedWindow(winname=WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, draw_circle_adv, param=img)


while True:
    # Draw window
    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
