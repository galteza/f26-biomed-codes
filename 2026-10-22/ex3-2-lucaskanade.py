import cv2
import numpy as np
import argparse

FLOW_W = 10
FLOW_H = 10

parser = argparse.ArgumentParser(description="Taking the parameters for the Lucas-Kanade method")
parser.add_argument("file1", help="Import first file for procedure")
parser.add_argument("file2", help="Import second file for procedure")

args = parser.parse_args()

img_pre = cv2.imread(args.file1)
img_pre = cv2.resize(img_pre, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
img_now = cv2.imread(args.file2)
img_now = cv2.resize(img_now, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

if img_pre.shape[:2] != img_now.shape[:2]:
    img_now = cv2.resize(img_now, (img_pre.shape[1], img_pre.shape[0]))


img_pre_g = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
img_now_g = cv2.cvtColor(img_now, cv2.COLOR_BGR2GRAY)

rows, cols, ch = img_now.shape

ps = np.empty((0,2), np.float32)

for y in range(0,cols,FLOW_H):
    for x in range(0,rows,FLOW_H):
        pp = np.array([[y,x]], np.float32)
        ps = np.vstack((ps,pp))

pe, status, error = cv2.calcOpticalFlowPyrLK(img_pre_g, img_now_g, ps, None)

for i in range(len(ps)):
    pt1 = int(ps[i][0]), int(ps[i][1])
    pt2 = int(pe[i][0]), int(pe[i][1])

    cv2.line(
        img_now,
        pt1, # (ps[i][0], ps[i][1]),
        pt2, # (pe[i][0], pe[i][1]),
        (0,0,255),
        1,
        )

cv2.putText(img_now,"282658015",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),5)



cv2.imshow('opticalflow', img_now)
cv2.waitKey(0)
cv2.destroyAllWindows()