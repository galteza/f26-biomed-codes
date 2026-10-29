import cv2
import numpy as np

h_upper,h_lower=115,60
s_upper,s_lower=255,50
v_upper,v_lower=200,20
win="main"

cap=cv2.VideoCapture("Toe_control.mp4")
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

    cv2.circle(img_src,pos,5,(0,0,255),-1)
    cv2.circle(img_src,(int(prediction[0]),int(prediction[1])),5,(0,255,255),-1)
    cv2.ellipse(img_src,(int(prediction[0]),int(prediction[1])),(abs(int(prediction[2])),abs(int(prediction[3]))),0,0,360,(0,255,255),3)

    cv2.imshow(win,img_src)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()