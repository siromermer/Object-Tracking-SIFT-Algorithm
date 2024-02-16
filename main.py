import cv2 
import numpy as np
import matplotlib.pyplot as plt 



# path to video  
video_path=r"C:\Users\sirom\OneDrive\Masaüstü\SIFT_ObjectTracking\resources\plane (1).mp4"   
video = cv2.VideoCapture(video_path)

# read only first frame for drawing rectangle for desired object
ret,frame = video.read()

#  i am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
x_min,y_min,x_max,y_max=36000,36000,0,0


def coordinat_chooser(event,x,y,flags,param):
    global go , x_min , y_min, x_max , y_max

    # when you click right button it is gonna give variables some coordinates
    if event==cv2.EVENT_RBUTTONDOWN:
        
        # if current coordinate of x lower than the x_min it will be new x_min , same rules apply for y_min 
        x_min=min(x,x_min) 
        y_min=min(y,y_min)

         # if current coordinate of x higher than the x_max it will be new x_max , same rules apply for y_max
        x_max=max(x,x_max)
        y_max=max(y,y_max)

        # draw rectangle
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),1)


    """
        if you didnt like your rectangle (maybe if you did some misscliks) ,  reset coordinates with middle button of your mouse
        if you press middle button of your mouse coordinate will reset and you can give new 2 point pair for your rectangle
    """
    if event==cv2.EVENT_MBUTTONDOWN:
        print("reset coordinate  data")
        x_min,y_min,x_max,y_max=36000,36000,0,0

cv2.namedWindow('coordinate_screen')
# Set mouse handler for the specified window , in this case "coordinate_screen" window
cv2.setMouseCallback('coordinate_screen',coordinat_chooser)


while True:
    cv2.imshow("coordinate_screen",frame) # show only first frame 
    
    k = cv2.waitKey(5) & 0xFF # after drawing rectangle press esc   
    if k == 27:
        cv2.destroyAllWindows()
        break


# take region of interest
roi_image=frame[y_min:y_max,x_min:x_max]
# convert roi to gray scale 
roi_gray=cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY) 


# path to video  
video_path=r"C:\Users\sirom\OneDrive\Masaüstü\SIFT_ObjectTracking\resources\plane (1).mp4"  
video = cv2.VideoCapture(video_path)

# matcher object
bf = cv2.BFMatcher()
 

 
# create SIFT algorithm object
sift = cv2.SIFT_create()

# find roi's keypoints and descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(roi_gray, None)

roi_keypoint_image=cv2.drawKeypoints(roi_gray,keypoints_1,roi_gray)

while True :
    ret,frame=video.read()

    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray, None)

    matches =bf.match(descriptors_1, descriptors_2)

    for match in matches:
        
        
        # from roi
        query_idx = match.queryIdx
        # current frame
        train_idx = match.trainIdx
        
        pt1 = keypoints_1[query_idx].pt
        # current frame keypoints
        pt2 = keypoints_2[train_idx].pt

        cv2.circle(frame,(int(pt2[0]),int(pt2[1])),3,(255,0,0),3)


    cv2.imshow("coordinate_screen",frame) # show only first frame 
 

    k = cv2.waitKey(5) & 0xFF # after drawing rectangle press esc   
    if k == 27:
        cv2.destroyAllWindows()
        break
        
cv2.destroyAllWindows()