import cv2

capture = cv2.VideoCapture(0)


def draw_rectangle(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset rect
        if state["top_clicked"] and state["bottom_clicked"]:
            state = {
                "p1": (0, 0),
                "p2": (0, 0),
                "top_clicked": False,
                "bottom_clicked": False,
            }
        if not state["top_clicked"]:
            state["p1"] = (x, y)
            state["top_clicked"] = True
        elif not state["bottom_clicked"]:
            state["p2"] = (x, y)
            state["bottom_clicked"] = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        state = {
            "p1": (0, 0),
            "p2": (0, 0),
            "top_clicked": False,
            "bottom_clicked": False,
        }


state = {
    "p1": (0, 0),
    "p2": (0, 0),
    "top_clicked": False,
    "bottom_clicked": False,
}

cv2.namedWindow("video")
cv2.setMouseCallback("video", draw_rectangle)

while True:
    ret, frame = capture.read()

    if state["top_clicked"]:
        cv2.circle(
            frame,
            center=state["p1"],
            radius=5,
            color=(255, 255, 255),
            thickness=-1,
        )
    if state["top_clicked"] and state["bottom_clicked"]:
        cv2.rectangle(
            frame,
            state["p1"],
            state["p2"],
            (255, 255, 255),
            3,
        )

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyWindow("video")
