import cv2
from pprint import pprint
from typing import Tuple


def ask_for_tracker() -> Tuple[str, cv2.Tracker]:
    print(
        """
    Choose Tracker:
           0. Boosting
           1. MIL
           2. KCF
           3. TLD
           4. MedianFlow
    Any other number to quit.
    """
    )
    choice = input("Please select your tracker: ").strip()[0]
    trackers = [
        cv2.legacy.TrackerBoosting,
        cv2.TrackerMIL,
        cv2.TrackerKCF,
        cv2.legacy.TrackerTLD,
        cv2.legacy.TrackerMedianFlow,
    ]
    if choice.isdigit():
        choice = int(choice)

        if choice >= len(trackers):
            print("Bye!")
            exit(0)
        else:
            tracker = trackers[choice].create()
            tracker_name = str(tracker).split()[0][1:]

            return tracker_name, tracker
    else:
        pprint("Invalid input\n")
        return ask_for_tracker()


name, tracker = ask_for_tracker()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Capture not open.")
    exit(-1)

ret, frame = cap.read()

roi = cv2.selectROI("Select ROI", frame, False)

ret = tracker.init(frame, roi)

while True:
    ret, frame = cap.read()

    success, roi = tracker.update(frame)

    (x, y, w, h) = tuple(map(int, roi))

    if success:
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    else:
        cv2.putText(
            frame,
            "Failure to detect tracking!",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.putText(frame, name, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow(name, frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
