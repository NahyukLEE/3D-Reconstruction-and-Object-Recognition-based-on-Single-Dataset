import cv2

vidcap = cv2.VideoCapture('KakaoTalk_Video_20200310_1626_33_406.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("newauto/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1