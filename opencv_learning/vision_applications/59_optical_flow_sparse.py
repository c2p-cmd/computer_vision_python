import cv2
import numpy as np

corner_track_params = {
    "maxCorners": 10,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7,
}
lk_params = {
    "winSize": (200, 200),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise PermissionError("Capture not running!")

ret, previous_frame = cap.read()
previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Points to track
prev_points = cv2.goodFeaturesToTrack(
    previous_frame_gray,
    mask=None,
    **corner_track_params,
)

mask = np.zeros_like(previous_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        previous_frame_gray,
        frame_gray,
        prev_points,
        np.zeros((2,)),
        **lk_params,
    )

    good_new = next_points[status == 1]
    good_prev = prev_points[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = np.ravel(new).astype(int)
        x_prev, y_prev = np.ravel(prev).astype(int)

        mask = cv2.line(
            mask,
            pt1=(x_new, y_new),
            pt2=(x_prev, y_prev),
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)

    img = cv2.add(frame, mask)
    cv2.imshow("Tracking", img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):
        break

    previous_frame_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)


cap.release()
cv2.destroyAllWindows()
