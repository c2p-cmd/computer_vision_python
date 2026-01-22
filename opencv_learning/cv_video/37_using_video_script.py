import cv2
import time

capture = cv2.VideoCapture('first_video.mp4')

if capture.isOpened() is False:
    print("File not found")
    exit(-1)

while capture.isOpened():
    ret, frame = capture.read()

    if ret:
        time.sleep(1/30)
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
        
capture.release()
cv2.destroyAllWindows()