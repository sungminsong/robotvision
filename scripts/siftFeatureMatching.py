import cv2 
import matplotlib.pyplot as plt
import time
import numpy as np

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2)


cap = cv2.VideoCapture('../data/road_vid.mp4',cv2.IMREAD_GRAYSCALE)
_, original = cap.read()
x,y,w,h	= cv2.selectROI('orignal', original, False)
if w and h:
    roi = cap[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)  # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('cropped', 0, 0) # 새창을 화면 좌측 상단에 이동
    cv2.imwrite('./cropped2.jpg', roi)   # ROI 영역만 파일로 저장
    img2 = roi.copy()
    
cv2.waitKey(0)   
cv2.destroyAllWindows()


while cap.isOpened():
    # read images

    suc, img1 = cap.read()
    #img2 = img1
    #img1 = cv2.imread('pencil3.jpg')  
    #img2 = cv2.imread('pencil3.jpg') 

    start = time.time()

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)


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
            pts2.append(keypoints_2[m.trainIdx].pt)
            pts1.append(keypoints_1[m.queryIdx].pt)
        
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    
    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
  
  
    mtrx, mask = cv2.findHomography(pts2,pts1)
    
    
    h, w = img2.shape
    pts = np.float32([[[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]]])
    dst = cv2.perspectiveTransform(pts,mtrx)
    img1 = cv2.polylines(img1,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    x_left = (dst[0,0,0])
    x_right = (dst[3,0,0])
    width = x_right-x_left
    distance =0.1*(900/width)
    print(distance)
    
    
    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    #print("FPS: ", fps)
    
    

    # draw matching
    img3 = cv2.drawMatchesKnn(img1,keypoints_1,img2,keypoints_2,matches,None,**draw_params)
    cv2.putText(img3, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('SIFT', img3)


    if cv2.waitKey(16) & 0xFF == 27:
        break


cap.release()