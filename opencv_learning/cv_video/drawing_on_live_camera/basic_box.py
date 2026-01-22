import cv2

capture = cv2.VideoCapture(0)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Dimensions: ", (width, height))

x = width // 2
y = height // 2

w = width // 4
h = height // 4

# Bottom right x+w, y+h

while True:
    ret, frame = capture.read()

    frame = cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color=(127, 0, 255),
        thickness=4,
    )
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyWindow("frame")
