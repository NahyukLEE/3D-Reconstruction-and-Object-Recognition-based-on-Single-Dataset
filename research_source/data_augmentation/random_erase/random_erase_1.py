import os
from os import listdir
from os.path import isfile, join
import natsort

import cv2
import random

#Image&Label Load
img_folder = r'C:\Users\CGlab\Desktop\random_test\img'
img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
img_files = natsort.natsorted(img_files, reverse=False)

label_folder = r'C:\Users\CGlab\Desktop\random_test\label'
label_files = [f for f in listdir(label_folder) if isfile(join(label_folder, f))]
label_files = natsort.natsorted(label_files, reverse=False)

for i in range(0, len(img_files)) :
    os.chdir(img_folder)
    img = cv2.imread(img_files[i])

    os.chdir(label_folder)
    label = open(label_files[i], 'r')
    r = label.readlines()
    ll = r[0].split()

    x1 = int(float(ll[1]) * img.shape[1] - float(ll[3]) * img.shape[1] / 2)
    y1 = int(float(ll[2]) * img.shape[0] - float(ll[4]) * img.shape[0] / 2)
    x2 = int(float(ll[1]) * img.shape[1] + float(ll[3]) * img.shape[1] / 2)
    y2 = int(float(ll[2]) * img.shape[0] + float(ll[4]) * img.shape[0] / 2)

    #img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    while True:
        weight = 1
        x_c = random.randint(int(x1 + (1 - weight) * x2), int(weight * x2))
        y_c = random.randint(int(y1 + (1 - weight) * y2), int(weight * y2))
        w = random.randint(int(0.1 * (img.shape[1] + img.shape[0]) / 2), int(0.5 * (img.shape[1] + img.shape[0]) / 2))
        h = random.randint(int(0.1 * (img.shape[1] + img.shape[0]) / 2), int(0.5 * (img.shape[1] + img.shape[0]) / 2))

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

        s = (x2 - x1) * (y2 - y1)
        d = width * height

        #print(d / s)
        if d / s > 0.1 and d / s < 0.2:
            img = cv2.rectangle(img, (x3, y3), (x4, y4), (190, 190, 190), -1)
            break

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()