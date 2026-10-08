import cv2
import argparse

parser = argparse.ArgumentParser(description="Run border detection on selected image file")
parser.add_argument("file",help="Insert file path")

args = parser.parse_args()

img_src = cv2.imread(args.file)
img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
img_dst = img_src.copy()

corners = cv2.cornerHarris(img_gray,3,3,0.02)

img_dst[corners>0.01*corners.max()] = [0,0,255]

cv2.imshow('src',img_src)
cv2.imshow('dst',img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()