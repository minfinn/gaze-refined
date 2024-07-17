import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image

import warp_norm
import warp_norm_old
import pickle
import test
def get_face_center_label(x, y, image_center):
    # # image_center = (640 / 2, 480 / 2)
    # distance_to_center = ((x - image_center[0]) ** 2 + (y - image_center[1]) ** 2) ** 0.5
    # # print(distance_to_center)
    # if distance_to_center <= 58:
    #     return 0 #upright
    # else:
    #     # 判断坐标点所在的区域
    #     diagonal1 = (image_center[1] / image_center[0]) * x
    #     diagonal2 = 2*image_center[1] - (image_center[1] / image_center[0]) * x
    #
    #     if y <= image_center[1] and y <= diagonal1 and y <= diagonal2:#上
    #         return 1
    #     elif x > image_center[0] and y < diagonal1 and y > diagonal2:#右
    #         return 2
    #     elif y > image_center[1] and y >= diagonal1 and y >= diagonal2:#下
    #         return 3
    #     elif x < image_center[0] and y > diagonal1 and y < diagonal2:#左
    #         return 4
    return 0


cam_chen = 'D:/Deeplearning/dataset/Calibration/Calibration/camChen.xml'  # this is camera calibration information file obtained with OpenCV
fs_chen = cv2.FileStorage(cam_chen, cv2.FILE_STORAGE_READ)
w_chen = 1300
h_chen = 720
pixel_scale_chen = 0.223427
camera_matrix_chen = fs_chen.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_chen = fs_chen.getNode('Distortion_Coefficients').mat()
cam_tan = 'D:/Deeplearning/dataset/Calibration/Calibration/camTan.xml'  # this is camera calibration information file obtained with OpenCV
fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
w_tan = 1600
h_tan = 825
pixel_scale_tan = 0.211667
camera_matrix_tan = fs_tan.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()
def get_camera(path):
    """video_test"""
    # return camera_matrix_tan, camera_distortion_tan, w_tan, h_tan, pixel_scale_tan
    return camera_matrix_chen, camera_distortion_chen, w_chen, h_chen, pixel_scale_chen


    name, extension = os.path.splitext(path)
    number = ''.join(filter(str.isdigit, name))
    number = int((int(number) - 1)/100)
    if(number % 2 == 0):
        return camera_matrix_tan, camera_distortion_tan, w_tan, h_tan, pixel_scale_tan
    else:
        return camera_matrix_chen, camera_distortion_chen, w_chen, h_chen, pixel_scale_chen


# 图像文件所在的文件夹路径
image_folder_path = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Photo'

# 预处理后的数据存储路径
save_dir = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/preprocessed_images'
os.makedirs(save_dir, exist_ok=True)

# 数据集列表
dataset = []

# 标签列表
load_labels = []
with open(os.path.join('D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass', 'coordinate.txt'), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # print(row)
        load_labels.append(row)
# gaze_centers =[[int(i[-2]), int(i[-1])] for i in load_labels[0:]]
gaze_centers =[[int(i[-3]), int(i[-2])] for i in load_labels[0:]]

# print(gaze_centers)

model1, model2, model3 = warp_norm.xmodel()
# 遍历图像文件夹
for filename in sorted(os.listdir(image_folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    if filename.endswith(".jpg"):
        # 构建图像文件的完整路径
        image_path = os.path.join(image_folder_path, filename)
        print(image_path)
        label = np.array(gaze_centers[int(''.join(filter(str.isdigit, filename))) - 1])
        # print(label)
        camera_matrix, camera_distortion, w, h, pixel_scale = get_camera(image_path)
        # 读取图像
        image = cv2.imread(image_path)
        # print(h)
        image, gaze_center, R, Ear, face_center_in_img = warp_norm.GazeNormalization(image, camera_matrix,camera_distortion,label,w,h,predictor=model1, face_detector=model2, eve_detector=model3)
        if(Ear == -1):
            continue
        # 保存预处理后的图像
        scale=np.array([[1,1,1],[1,1,1],[0.8,0.8,0.8]])
        R = R * scale
        # print(R)
        # 保存预处理后的图像
        save_path = os.path.join(save_dir, f'preprocessed_image_{filename}')
        cv2.imwrite(save_path, image)
        print(face_center_in_img)
        # 添加到数据集列表
        face_area_label = get_face_center_label(face_center_in_img[0], face_center_in_img[1], (320, 240))
        print(face_area_label)
        dataset.append({'image_path': f'preprocessed_image_{filename}', 'original_label': label, 'R': R, 'face_area_label':face_area_label})


pickle_file_path = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/dataset_dict.pkl'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(dataset, file)

# 保存标签为CSV
csv_file_path = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/preprocessed_labels.csv'
df = pd.DataFrame(dataset)
df.to_csv(csv_file_path, index=False)

print('Preprocessing and saving complete.')