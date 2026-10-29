import cv2
import numpy as np

img_src = cv2.imread('sunflower_1.jpg', 1)
img_template = cv2.imread('sunflower_template.jpg', 1)

img_dst = img_src.copy()

h,w,ch=img_template.shape
img_minmax=cv2.matchTemplate(img_src,img_template,cv2.TM_CCOEFF_NORMED)

cv2.rectangle(img_dst,max_pt,(max_pt[0]+w,max_pt[1]+h),10,(0,0,0))

cv2.imshow(img_src)
cv2.imshow(img_template)
cv2.imshow(img_minmax)
cv2.imshow(img_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
