import cv2
import numpy as np

image = np.zeros((480, 640, 3), np.uint8)

cv2.line(image,(0,0), (640,480), (0, 255, 255), 5)
cv2.circle(image,(320, 240), 100, (255, 0, 0), 3)
cv2.circle(image, (500, 100), 50, (0, 255, 0), -1)
cv2.rectangle(image, (100,150), (150, 300), (0,0,255), 2)
cv2.rectangle(image,(50,350),(250,400),(255,0,255),-1)
cv2.putText(image,"123",(300,450),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),5)
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()