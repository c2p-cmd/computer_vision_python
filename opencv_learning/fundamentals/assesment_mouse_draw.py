import cv2
import numpy as np

WINDOW_NAME = "Circle Drawing"


def draw_circle(event, x, y, flags, param):
    if event != cv2.EVENT_RBUTTONDOWN:
        return
    cv2.circle(
        param,
        radius=25,
        center=(x, y),
        color=(0, 0, 255),
        thickness=3,
    )


img = np.zeros(shape=(512, 768, 3))

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, draw_circle, img)

while True:
    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
