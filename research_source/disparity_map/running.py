import numpy as np
import cv2
from matplotlib import pyplot as plt

video_file = "SampleVideo_1280x720_30mb.mp4"
cap = cv2.VideoCapture(video_file)

cap.set(3,320)
cap.set(4,320)

fps = cap.get(cv2.CAP_PROP_FPS)
delay=int(1000/fps)
print(fps)
print(delay)

while True :
    ret, imgL = cap.read()
    cv2.waitKey(10*delay)
    ret, imgR = cap.read()
    
    imgL_new = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_new = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL_new,imgR_new, cv2.CV_32F)

    res = cv2.convertScaleAbs(disparity)

    
    cv2.imshow('depth', res)
    cv2.imshow('imgL', imgL_new)
    cv2.imshow('imgR', imgR_new)
    #plt.imshow(disparity,'gray')
    #plt.show()

    
cap.release()
cv2.destroyAllWindows()
