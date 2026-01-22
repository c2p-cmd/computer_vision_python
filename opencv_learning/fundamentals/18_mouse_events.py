import cv2
import numpy as np

WINDOW_NAME = "My Drawing"

state = {
    "is_drawing": False,
    "x": 0,
    "y": 0,
}


def draw_rect(event, x, y, flags, param):
    global state

    if event == cv2.EVENT_LBUTTONDOWN:
        state = {"is_drawing": True, "x": x, "y": y}
    elif event == cv2.EVENT_MOUSEMOVE:
        if state["is_drawing"]:
            cv2.rectangle(param, (state["x"], state["y"]), (x, y), (0, 255, 200), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        state["is_drawing"] = False
        cv2.rectangle(param, (state["x"], state["y"]), (x, y), (0, 255, 200), -1)


img = np.zeros((768, 512, 3))

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, draw_rect, param=img)

while True:
    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
