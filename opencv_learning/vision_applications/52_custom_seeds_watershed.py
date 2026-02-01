import cv2
import numpy as np
from matplotlib import colormaps
from pprint import pprint
from dataclasses import dataclass

road = cv2.imread("../../resources/DATA/road_image.jpg")

assert road is not None, "Image wasn't read!"

road_copy = road.copy()

pprint(f"Image shape: {road.shape}")

marker_image = np.zeros(road.shape[:2], dtype=np.int32)
segments = np.zeros_like(road, dtype=np.uint8)


def create_rgb_tuple(i):
    return tuple([255 * c for c in colormaps.get_cmap("tab10").colors[i]])


@dataclass
class State:
    colors: list[tuple[int, int, int]]
    current_marker: int = 1
    marks_updates: bool = False
    n_markers: int = 10


state = State(
    colors=[create_rgb_tuple(i) for i in range(10)],
)


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global state

    if event == cv2.EVENT_LBUTTONDOWN:
        # markers sent to watershed algorithm
        cv2.circle(marker_image, (x, y), 10, (state.current_marker), -1)

        # user sees on the road image's copy
        cv2.circle(road_copy, (x, y), 10, state.colors[state.current_marker], -1)

        state.marks_updates = True


window_name = "Road Image"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    cv2.imshow("Watershed Segments", segments)
    cv2.imshow(window_name, road_copy)

    k = cv2.waitKey(1)

    if k == 27 or k == ord("q"):
        break
    elif k == ord("c"):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros_like(road, dtype=np.uint8)
    elif k > 0 and chr(k).isdigit():
        state.current_marker = int(chr(k))

    if state.marks_updates:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        segments = np.zeros_like(road, dtype=np.uint8)

        for color_idx in range(state.n_markers):
            segments[marker_image_copy == color_idx] = state.colors[color_idx]


cv2.destroyAllWindows()
