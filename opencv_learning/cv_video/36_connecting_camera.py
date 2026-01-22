import cv2

capture = cv2.VideoCapture(0)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Dimensions: ", (width, height))

writer = cv2.VideoWriter('first_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 24, (width, height))

while True:
    ret, frame = capture.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    writer.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
capture.release()
cv2.destroyWindow('frame')