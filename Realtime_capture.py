import numpy as np
import cv2
import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import cv2
import warp_norm
import matplotlib
import sys
sys.path.append("./FaceAlignment")
import face_alignment
from skimage import io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import gaze_network
from torchvision import transforms
import utils
import pandas as pd
import copy
from pt_module import StNet,StRefine
from ipdb import set_trace as st
import refine
import warp_norm
import warp_norm_old
import pickle
import data_loader
import trainer
import argparse
from config import get_config
cap = cv2.VideoCapture(0)
def get_camera():
    cam_tan = 'D:/Deeplearning/dataset/Calibration/Calibration/camTan.xml'  # this is camera calibration information file obtained with OpenCV
    fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
    w_tan = 2560
    h_tan = 1600
    pixel_scale_tan = 0.211667
    camera_matrix_tan = fs_tan.getNode(
        'Camera_Matrix').mat()  # camera calibration information is used for data normalization
    camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()
    return camera_matrix_tan, camera_distortion_tan, w_tan, h_tan, pixel_scale_tan

def get_face_center_label(x, y, image_center):
    # image_center = (640 / 2, 480 / 2)
    distance_to_center = ((x - image_center[0]) ** 2 + (y - image_center[1]) ** 2) ** 0.5
    # print(distance_to_center)
    if distance_to_center <= 58:
        return 0 #upright
    else:
        # 判断坐标点所在的区域
        diagonal1 = (image_center[1] / image_center[0]) * x
        diagonal2 = 2*image_center[1] - (image_center[1] / image_center[0]) * x

        if y <= image_center[1] and y <= diagonal1 and y <= diagonal2:#上
            return 1
        elif x > image_center[0] and y < diagonal1 and y > diagonal2:#右
            return 2
        elif y > image_center[1] and y >= diagonal1 and y >= diagonal2:#下
            return 3
        elif x < image_center[0] and y > diagonal1 and y < diagonal2:#左
            return 4


config, unparsed = get_config()
config.is_train = False
config.batch_size = 1
config.use_gpu = True
config.ckpt_dir = './ckpt'
config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
colors = plt.cm.viridis(np.linspace(0, 1, 4))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

model_dir = 'D:/Deeplearning/GazeNormalization-main/models'
state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
state_path = model_dir + '/' + state_name


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('realtime_vedio/output.avi',fourcc, 20.0, (224,224))
model1, model2, model3 = warp_norm.xmodel()
filename= 0
save_dir = 'realtime_dataset'
os.makedirs(save_dir, exist_ok=True)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        """预处理"""
        camera_matrix, camera_distortion, w, h, pixel_scale = get_camera()
        # 读取图像
        # print(h)
        image, gaze_center, R, Ear, face_center_in_img = warp_norm.GazeNormalization(frame, camera_matrix,
                                                                                     camera_distortion,np.array([100, 100]), w, h,
                                                                                     predictor=model1,
                                                                                     face_detector=model2,
                                                                                     eve_detector=model3)
        if (Ear == -1):
            continue
        scale = np.array([[1, 1, 1], [1, 1, 1], [0.8, 0.8, 0.8]])
        R = R * scale
        # print(R)
        # 保存预处理后的图像
        save_path = os.path.join(save_dir, f'preprocessed_image_{filename}.jpg')
        cv2.imwrite(save_path, image)
        # print(face_center_in_img)
        # 添加到数据集列表
        face_area_label = get_face_center_label(face_center_in_img[0], face_center_in_img[1], (320, 240))
        # print(face_area_label)
        dataset = []
        dataset.append({'image_path': f'preprocessed_image_{filename}.jpg', 'original_label': np.array([100, 100]), 'R': R,
                        'face_area_label': face_area_label})

        pickle_file_path = 'realtime_dataset/dataset_dict.pkl'
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(dataset, file)

        # 保存标签为CSV
        csv_file_path = 'realtime_dataset/preprocessed_labels.csv'
        df = pd.DataFrame(dataset)
        df.to_csv(csv_file_path, index=False)




        """预测"""
        test_data = data_loader.get_test_loader('realtime_dataset/preprocessed_labels.csv',
                                                batch_size=config.batch_size,
                                                data_path='realtime_dataset',
                                                num_workers=0)
        xgaze = trainer.Trainer(config, test_data)
        xgaze.test(save_fname_pkl='realtime_dataset/gaze_pred.pkl', save_fname_txt='realtime_dataset/results.txt')

        """可视化"""
        pred = []
        pred_path = 'realtime_dataset/results.txt'
        with open(pred_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            data_array = [float(x) for x in line.split()]
            pred.append(data_array)
        pred = np.array(pred)

        pickle_file_path = 'realtime_dataset/dataset_dict.pkl'
        with open(pickle_file_path, 'rb') as file:
            loaded_dataset_dict = pickle.load(file)

        image_path = 'realtime_dataset'
        save_path = 'realtime_dataset/visualize'
        os.makedirs(save_path, exist_ok=True)
        idx = 0
        for data in loaded_dataset_dict:
            image_name = os.path.basename(data['image_path'])
            print(image_name)
            ground_truth = np.array([data['original_label'].reshape((1, 2))[0]])
            img = cv2.imread(os.path.join(image_path, image_name))
            warp_norm.draw_gaze(img, pred[idx])
            cv2.imwrite(os.path.join(save_path, image_name), img)

        """refine"""


        def Revert_normalization(pred_pitchyaw, RMat):
            pred_vector = warp_norm.pitchyaw_to_vector(pred_pitchyaw)
            org_pred=(np.dot(np.linalg.inv(RMat), pred_vector.T))
            return org_pred

        RMat=R
        pred_gc_org = Revert_normalization(pred, RMat)
        print(pred_gc_org)






        """保存history"""
        out.write(image)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
