import cv2
import argparse

parser = argparse.ArgumentParser(description="Run Canny Edge corner detection on selected file")
parser.add_argument("file", help="Import image file")

args = parser.parse_args()

img_src = cv2.imread(args.file)
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_dst = cv2.Canny(img_gray, 300, 600)

cv2.imshow('src', img_src)
cv2.imshow('dst', img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()