import cv2
import random

img = cv2.imread('frame0.jpg')

label = open('frame0.txt', 'r')

r = label.readlines()
ll = r[0].split()

x1 = int(float(ll[1]) * img.shape[1] - float(ll[3]) * img.shape[1] / 2)
y1 = int(float(ll[2]) * img.shape[0] - float(ll[4]) * img.shape[0] / 2)
x2 = int(float(ll[1]) * img.shape[1] + float(ll[3]) * img.shape[1] / 2)
y2 = int(float(ll[2]) * img.shape[0] + float(ll[4]) * img.shape[0] / 2)

print(img.shape)

img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

print(img.shape)

while True :
    weight = 1
    x_c = random.randint(int(x1 + (1-weight) * x2), int(weight * x2))
    y_c = random.randint(int(y1 + (1-weight) * y2), int(weight * y2))
    w = random.randint(int(0.1 * (img.shape[1]+img.shape[0]) / 2), int(0.5 * (img.shape[1]+img.shape[0]) / 2))
    h = random.randint(int(0.1  * (img.shape[1]+img.shape[0]) / 2), int(0.5 * (img.shape[1]+img.shape[0]) / 2))

    x3 = int(x_c - w / 2)
    y3 = int(y_c - h / 2)
    x4 = int(x_c + w / 2)
    y4 = int(y_c + h / 2)

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height = right_down_y - left_up_y

    s = (x2-x1)*(y2-y1)
    d = width * height

    print(d/s)
    if d/s > 0.1 and d/s < 0.2 :
        img = cv2.rectangle(img, (x3, y3), (x4, y4), (190, 190, 190), -1)
        break

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
