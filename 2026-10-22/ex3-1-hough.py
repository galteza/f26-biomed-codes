import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Takes the parameters for doing hough transform")
parser.add_argument("file", help="Import image file")
args = parser.parse_args()


img_src = cv2.imread(args.file)
img_dst = img_src.copy()
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

img_edge = cv2.Canny(img_gray, 150, 200)
lines = cv2.HoughLines(img_edge, 1, np.pi/180, 200)
rows, cols = img_dst.shape[:2]

for rho, theta in lines[:,0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    cv2.line(
        img_dst,
        (int(x0-cols*(b)), int(y0+cols*(a))),
        (int(x0+cols*(b)), int(y0-cols*(a))),
        (0,0,255),
        1,
        )

cv2.imshow('src', img_src)
cv2.imshow('edge', img_edge)
cv2.imshow('dst', img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
    
