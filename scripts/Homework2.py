import cv2
import numpy as np
import copy

points_src = []

def mouse_event_handler(event, x, y, flags, param):
    global points_src
    if event == cv2.EVENT_LBUTTONDOWN:
        points_src.append((x,y))
        # points_src = np.append(points_src, np.array((x,y)))

def mouse_event(x,y,w,h):
    x,y,w,h = cv2.selectROI('img', img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        cv2.imshow('cropped', roi)                   # ROI 지정 영역을 새창으로 표시
        cv2.moveWindow('cropped', 0, 0)              # 새창을 화면 측 상단으로 이동
        cv2.imwrite('../CV2/img/cropped2.jpg', roi)  # ROI 영역만 파일로 저장
    
def main():
    global points_src
    cap = cv2.VideoCapture('../data/road_vid.mp4')
    window_name = "Bird-Eye-View"

    # card_size = np.array([450, 250])
    view_size = (360,720)

    # Prepare the rectified points
    points_dst = np.array([[0,0], [view_size[0],0], [0, view_size[1]], [view_size[0], view_size[1]]])

    # Load a video
    _, original = cap.read()    # image가 아닌 video를 가져옴
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event_handler) 
    while len(points_src) < 4: 
        display = copy.deepcopy(original)        
        idx = min(len(points_src), len(points_dst))
        if len(points_src) > 0:            
            display = cv2.circle(display, points_src[idx-1], 5, (0, 255, 0), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) == ord('q'): break
    
    cv2.destroyAllWindows() # 점을 마우스로 찍은 후 image를 파괴함
    
    while True:
        _, original = cap.read() # video를 가져옴
        
        # bird eye view를 할 좌표
        pts1 = np.float32([points_src[0],points_src[1],points_src[2],points_src[3]])
        
       # H = cv2.getPerspectiveTransform(pts1, points_dst)
        points_src = np.array(points_src, dtype=np.float32)
        H, inliner_mask = cv2.findHomography(pts1, points_dst, cv2.RANSAC)
        rectify = cv2.warpPerspective(original, H, view_size)
        
        # Show Image
        cv2.imshow("original img", original)
        cv2.imshow("bird-eye-view img", rectify)
        key = cv2.waitKey(1)
        
        if key == 27:
            break
  

cv2.destroyAllWindows() # 영상 조건 만족 후 video를 파괴함

if __name__ == '__main__' :
    main()
