import cv2

from random import randrange

train = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while (True):

    successfull_frame_read, frame = cam.read()

    gray_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cam_cood = train.detectMultiScale(gray_cam)

    for (x,y,w,h) in cam_cood:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('webcam',frame)
    cv2.waitKey(1)

print('code complete')