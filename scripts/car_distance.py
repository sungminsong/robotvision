import cv2 
import matplotlib.pyplot as plt
import numpy as np
import math
import time

sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2)


cap = cv2.VideoCapture('../data/car2.mp4',cv2.IMREAD_GRAYSCALE)
_, original = cap.read()
x,y,w,h	= cv2.selectROI('img', original, False)
if w and h:
    roi = original[y:y+h, x:x+w]
    cv2.imshow('car', roi)
    cv2.moveWindow('car', 0, 0)
    cv2.imwrite('./car.jpg', roi)
    img2 = roi.copy()
    
cv2.waitKey(0)   
cv2.destroyAllWindows()


while cap.isOpened():
    suc, img1 = cap.read()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    matches = bf.knnMatch(des1,des2,k=2)


    good = []
    pts1 = []
    pts2 = []
    
    # Need to draw only good matches, so create pts1_mean mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
        
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    
    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
  
  
    mtrx, mask = cv2.findHomography(pts2,pts1)
    
    h, w, c  = img2.shape
    pts = np.float32([[[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]]])
    dst = cv2.perspectiveTransform(pts,mtrx)
    img1 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    x_left = (dst[0,0,0])
    x_right = (dst[3,0,0])
    width = x_right-x_left
    distance =1.1*(1000/width)
    distance_ = math.floor(distance)
    

    # draw matching
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    

    cv2.putText(img3, f'distance: {int(distance_)} m', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2 )
    cv2.imshow('SIFT', img3)


    key = cv2.waitKey(500) & 0xFF
    if(key == 27):
        break


cap.release()
cv2.destroyAllWindows()