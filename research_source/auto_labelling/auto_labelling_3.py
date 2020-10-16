import os
from os import listdir
from os.path import isfile, join
import natsort

import cv2 
import numpy as np

import matplotlib.pyplot as plt

class_num = 0

# 이미지 데이터셋 경로 설정
img_folder = r'C:\Users\CGLab\Desktop\refrigerator_original\image'
os.chdir(img_folder)

# 레이블링 데이터 저장 경로 설정
label_save_folder = r'C:\Users\CGLab\Desktop\refrigerator_original\label'

# 이미지 경로 내 파일 목록 list 저장
img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]

# list 내 이미지 정보 오름차순 정렬
img_files = natsort.natsorted(img_files, reverse=False)

# 이미지 개수 카운팅
img_files_num = len(img_files)
count = 0

# 최소 Feature 매칭 카운팅 설정
MIN_MATCH_COUNT = 10

# 최소 Bounding Box 유사도 설정
MIN_AREA_SIMILARITY = 60

# 첫번째 프레임 이미지 불러오기
query_img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)

# 이미지 해상도 설정
query_img = cv2.resize(query_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR )

# 바운딩 박스 내 중심 격자 여부 설정
fromCenter = False

# 바운딩 박스 그리기
r = cv2.selectROI(query_img, fromCenter)

# 바운딩 박스 영역을 새로운 이미지 변수에 저장
img1 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]

print("----------------------------------")

width_data = []
height_data = []
feature_data = []
count_list = []

plt.figure()
    
sift_result = []
surf_result = []

# 모든 이미지에 대하여 아래 과정 반복
for img_name in img_files:

    # 바운딩 박스의 width와 height 정보를 받아옴
    h,w = img1.shape
    
    # 경로 설정
    os.chdir(img_folder)
    
    # 다음 프레임 이미지를 불러옴
    img2 = cv2.imread(img_name,0)
    
    # 이미지 해상도 설정
    #img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) ###
    
    print("Loading", img_name)
    
    # SIFT detector 로드
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    # SIFT 디스크립터로 특징점 탐색
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # KDTREE 인덱싱(SIFT 나 SURP의 경우 1)
    FLANN_INDEX_KDTREE = 1
    
    # 트리 개수 설정 (OpenCV 내 5 권장)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    
    # 검색 수
    search_params = dict(checks = 100)
    
    # KNN 매칭 수행, k순위 매칭 결과까지 리스트에 저장됨
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
 
    # n * n.distance 거리 내 매칭값 선정
    good = []
    for m,n in matches:
        if m.distance < 0.3 * n.distance :
            good.append(m)
    
    print("SIFT Match Points :", len(good), "(> 10)")
    sift_result.append(len(good))

    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    # KDTREE 인덱싱(SIFT 나 SURP의 경우 1)
    FLANN_INDEX_KDTREE = 1

    # 트리 개수 설정 (OpenCV 내 5 권장)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # 검색 수
    search_params = dict(checks=100)

    # KNN 매칭 수행, k순위 매칭 결과까지 리스트에 저장됨
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # n * n.distance 거리 내 매칭값 선정
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append(m)


    print("SURF Match Points :", len(good), "(> 10)")
    surf_result.append(len(good))



    # 최소 특징점 매칭 개수보다 많을 시
    if len(good)>MIN_MATCH_COUNT:
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # 호모그래피 계산
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        # 매칭된 왜곡 바운딩 박스를 그려줌
        cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2) 
        
        
        # 왜곡된 사각형의 네 꼭짓점 x 좌표를 리스트에 저장 후 오름차순 정렬
        x_list = [np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0]]
        x_list.sort()
        
        # 왜곡된 사각형의 네 꼭짓점 y 좌표를 리스트에 저장 후 오름차순 정렬
        y_list = [np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1]]
        y_list.sort()
        
        # 이미지 범위 밖 좌표 보정
        for i in range (0, 3, 1):
            if x_list[i] <= 0 :
                x_list[i] = 0
            if y_list[i] <= 0 :
                y_list[i] = 0
        
        
        # 새로운 바운딩 박스 그리기
        cv2.rectangle(img2, (x_list[0], y_list[0]), (x_list[3], y_list[3]), (0,255,0), 1)
        
        
        # YOLO 데이터셋을 위한 width, height, 중심 좌표 계산
        w = ( x_list[3] - x_list[0] ) / img2.shape[1]
        h = ( y_list[3] - y_list[0] ) / img2.shape[0]
        x = ((x_list[0] + x_list[3]) / 2 ) /img2.shape[1]
        y = ((y_list[0] + y_list[3]) / 2 ) /img2.shape[0]
        
        
        # YOLO 데이터셋 출력
        print("%d %.6f %.6f %.6f %.6f" % (class_num, x , y, w, h))
        
        
        # Matching Line 파라미터 설정
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        
        # Matching Line 그리기
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        
        # 이미지 매칭 최종 결과 출력
        cv2.imshow('img', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
        # 레이블링 그려지지 않은 깨끗한 이미지 다시 로드
        img2 = cv2.imread(img_name,0)
        
        # 해상도 설정
        #img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR ) ###
        
        # 새 바운딩 박스 영역 img1 변수에 넣어줌
        img1 = img2[y_list[0]:y_list[0] + ( y_list[3] - y_list[0] ), x_list[0]:x_list[0] + ( x_list[3] - x_list[0] )]
        
        
        # 경로 변경
        os.chdir(label_save_folder)
        
        # 이미지 이름 받아오기
        name = os.path.splitext(img_name)
        name = os.path.split(name[0])
        
        # 이미지 이름에 .txt 확장자 더하여 YOLO 레이블 저장
        label = open(name[1]+".txt", "w")
        label.write(str(class_num) + ' ' + str(round(x,6)) + ' ' + str(round(y,6)) + ' ' + str(round(w,6)) + ' ' + str(round(h,6)))
        
        
        print(x_list[3] - x_list[0], y_list[3] - y_list[0], len(good))
        
        width_data.append(x_list[3] - x_list[0])
        height_data.append(y_list[3] - y_list[0])
        feature_data.append(len(good))

        count_list.append(count)
        count = count + 1
        plt.plot(count_list, feature_data)
        

    
    # 충분한 특징점 매칭 결과가 찾아지지 않은 경우    
    else :
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    

    
    # esc를 누르면 종료
    if (cv2.waitKey(1) & 0xFF) == 27 : #esc
        cv2.destroyAllWindows()
        break

print('sift:', sift_result)
print('surf:', surf_result)

#for i in range(0, count, 1) :
    

