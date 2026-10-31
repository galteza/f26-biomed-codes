import cv2
import numpy as np
import argparse

"""
RGB to HSV conversion
1. Divide R, G, B by 255 to normalize them to the range [0, 1]
2. Compute the maximum and minimum of the normalized R, G, B values
3. Compute the difference (delta) between the maximum and minimum
4. V calculated as the maximum of the normalized R, G, B values
5. S calculated as 0 if V is 0, otherwise S = delta / V
6. H calculated as 0 if delta is 0, otherwise based on which color channel is the maximum
    If V = R, 60*((G-B)/delta mod 6)
    If V = G, 60*(B-R)/delta + 120
    If V = B, 60*(R-G)/delta + 240
7. H, S, V are then scaled to the ranges H: [0, 179], S: [0, 255], V: [0, 255] for OpenCV HSV representation
"""

RGB = (74, 96, 198)
RGB_normalized = tuple([x / 255.0 for x in RGB])
delta = max(RGB_normalized) - min(RGB_normalized)
V = max(RGB_normalized)
if V == 0:
    S = 0
else:
    S = delta / V
if delta == 0:
    H = 0
else:
    if V == RGB_normalized[0]:
        H = 60 * ((RGB_normalized[1] - RGB_normalized[2]) / delta % 6)
    elif V == RGB_normalized[1]:
        H = 60 * ((RGB_normalized[2] - RGB_normalized[0]) / delta + 2)
    elif V == RGB_normalized[2]:
        H = 60 * ((RGB_normalized[0] - RGB_normalized[1]) / delta + 4)

HSV = (int(H / 2), int(S * 255), int(V * 255))
print("HSV:", HSV)

# if not (0 < HSV[0] < 180) :
#     HSV[0] = HSV[0] % 180
# if not (0 < HSV[1] < 255):
#     HSV[1] = max(0, min(HSV[1], 255))
# if not (0 < HSV[2] < 255):
#     HSV[2] = max(0, min(HSV[2], 255))

parser = argparse.ArgumentParser(description="Kalman Filter Tracking")
parser.add_argument("--video", type=str, default="balloon.mp4", help="Path to the video file")
args = parser.parse_args()

# h_upper,h_lower=115,60
# s_upper,s_lower=255,50
# v_upper,v_lower=200,20

h_upper,h_lower=HSV[0]+10,HSV[0]-10
s_upper,s_lower=HSV[1]+50,HSV[1]-50
v_upper,v_lower=HSV[2]+50,HSV[2]-50


print("HSV Range:")
print("H:", h_lower, "-", h_upper)
print("S:", s_lower, "-", s_upper)
print("V:", v_lower, "-", v_upper)

win="main"

cap=cv2.VideoCapture(args.video)
img_src=None

KF=cv2.KalmanFilter(4,2)
KF.statePre=np.zeros((4,1),dtype=np.float32)
KF.transitionMatrix=np.array([[1,0,1,0],
                              [0,1,0,1],
                              [0,0,1,0],
                              [0,0,0,1]],dtype=np.float32)

cv2.setIdentity(KF.measurementMatrix)
cv2.setIdentity(KF.processNoiseCov,0.1)
cv2.setIdentity(KF.measurementNoiseCov,0.1)
cv2.setIdentity(KF.errorCovPost,0.1)

element8=np.ones((3,3),np.uint8)

while True:
    ret,img_src=cap.read()
    if not ret:
        break

    img_hsv=cv2.cvtColor(img_src,cv2.COLOR_BGR2HSV)
    img_bin=cv2.inRange(img_hsv,(h_lower,s_lower,v_lower),(h_upper,s_upper,v_upper))

    img_bin=cv2.erode(img_bin,element8,iterations=5)
    img_bin=cv2.dilate(img_bin,element8,iterations=5)

    num_labels,img_lbl,stats,centroids=cv2.connectedComponentsWithStats(img_bin)

    if num_labels ==1:
        continue

    max_area=0
    max_index=0
    for i in range(1,num_labels):
        area=stats[i,cv2.CC_STAT_AREA]
        if area > max_area:
            max_area=area
            max_index=1

    img_dst=np.zeros_like(img_bin)
    img_dst[img_lbl == max_index] = 255

    moments = cv2.moments(img_dst)
    if moments["m00"] != 0:
        pos=(int(moments["m10"]/moments["m00"]),int(moments["m01"]/moments["m00"]))
    else:
        pos=(0,0)

    measurement=np.array([[pos[0]],
                          [pos[1]]],dtype=np.float32)
    correction=KF.correct(measurement)
    prediction=KF.predict()

    cv2.circle(img_src,pos,15,(0,255,0),-1)
    cv2.circle(img_src,(int(prediction[0][0]),int(prediction[1][0])),5,(0,255,255),-1)
    cv2.ellipse(img_src,(int(prediction[0][0]),int(prediction[1][0])),(abs(int(prediction[2][0])),abs(int(prediction[3][0]))),0,0,360,(0,255,255),3)
    cv2.putText(img_src,"282658015",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),5)


    cv2.imshow(win,img_src)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()