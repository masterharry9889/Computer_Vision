import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('WhatsApp Image 2023-01-01 at 9.20.37 PM (1).jpeg')

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

face_cood = trained_face_data.detectMultiScale(gray_image)

for (x,y,w,h) in face_cood:
    # image, coordinates, color, thikness
    cv2.rectangle(image, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)),2)

print(face_cood)

cv2.imshow('detector',image)
cv2.waitKey()

print("code completed")