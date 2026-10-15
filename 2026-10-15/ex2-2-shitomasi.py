import cv2
import argparse

parser = argparse.ArgumentParser(description="Run Shi-Tomasi corner detection on selected file")
parser.add_argument("file", help="Insert file path")

args = parser.parse_args()

img_src = cv2.imread(args.file)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_dst = img_src.copy()

corners = cv2.goodFeaturesToTrack(img_gray, 1000, 0.1, 5)

for i in corners:
    x,y = i.ravel()
    x=int(x)
    y=int(y)
    cv2.circle(img_dst,(x,y),3,(0,0,255),2)

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()