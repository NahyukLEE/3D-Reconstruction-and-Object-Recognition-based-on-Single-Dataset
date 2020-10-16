import numpy as np
import cv2
 
#frame img input
img = cv2.imread('frame0.jpg')
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

#random background input
bg = cv2.imread('background_1.jpg')
bg = cv2.resize(bg, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

print('Loading original label data...', end = ' ')

label = open('frame0.txt', 'r')



r = label.readlines()
ll = r[0].split()

x1 = ( float(ll[1]) * img.shape[1] - float(ll[3]) * img.shape[1] / 2 )
y1 = ( float(ll[1]) * img.shape[1] - float(ll[4]) * img.shape[1] / 2 )
x2 = ( float(ll[1]) * img.shape[1] + float(ll[3]) * img.shape[1] / 2 )
y2 = ( float(ll[1]) * img.shape[1] + float(ll[4]) * img.shape[1] / 2 )
rect = (int(x1), int(y1), int(x2), int(y2))

print('Done')


print('Grabcut...', end = ' ')

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

print("Done")

cv2.imshow('mask', mask)

# 배경인 곳은 0, 그외에는 1로 설정한 마스크 제작
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# 배경인 곳은 1, 그외에는 0으로 설정한 마스크 제작
mask3 = np.where((mask==2)|(mask==0),1,0).astype('uint8')

# 이미지와 마스크 곱
image_rgb_nobg = img * mask2[:,:,np.newaxis]
sample = bg * mask3[:,:,np.newaxis]

cv2.imshow('result', image_rgb_nobg)
cv2.imshow('sample', sample)
cv2.imshow('aa', cv2.add(image_rgb_nobg, sample)) # 마스킹 된 두 이미지 합산

cv2.waitKey()
cv2.destroyAllWindows()
