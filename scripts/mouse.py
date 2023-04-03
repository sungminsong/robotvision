import cv2, numpy as np

img = cv2.imread('../data/road_img.jpg')

x,y,w,h = cv2.selectROI('img', img, False)
if w and h:
    roi = img[y:y+h, x:x+w]
    cv2.imshow('cropped', roi)                   # ROI 지정 영역을 새창으로 표시
    cv2.moveWindow('cropped', 0, 0)              # 새창을 화면 측 상단으로 이동
    cv2.imwrite('../CV2/img/cropped2.jpg', roi)  # ROI 영역만 파일로 저장
    
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()