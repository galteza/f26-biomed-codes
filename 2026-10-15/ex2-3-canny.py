import cv2
import argparse

parser = argparse.ArgumentParser(description="Run Canny Edge corner detection on selected file")
parser.add_argument("file", help="Import image file")

args = parser.parse_args()

img_src = cv2.imread(args.file)
img_src = cv2.resize(img_src, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_gray = cv2.resize(img_gray, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
img_dst = cv2.Canny(img_gray, 300, 600)
cv2.putText(img_dst,"282658015",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),5)


cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()