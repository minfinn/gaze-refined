import os.path
import random
import sys
import argparse
from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import re
import copy
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from itertools import combinations
import collections
# room_root = 'D:\\Deeplearning'
# sys.path.append(room_root)
# from utilities import generate_outlier_mark
import pandas as pd
from scipy.stats import multivariate_normal
from math import log
from ipdb import set_trace as st


def mark_saccade_and_blink(pupil_trail, blink_bools):
    event_labels = []
    pupil_velocity = np.zeros(pupil_trail.shape)
    pupil_velocity[1:] = pupil_trail[1:] - pupil_trail[:-1]#diff -> velocity
    pupil_acc = np.zeros(pupil_trail.shape)
    pupil_acc[1:] = pupil_velocity[1:] - pupil_velocity[:-1]#diff -> acc

    N = len(pupil_trail)
    event_labels = ['na'] * N

    # 1. mark velocity outliers(2 std) as saccade
    def mark_true_as(labels, bools, target_label, prlf_len=0):
        N = len(labels)
        for i in range(N):
            if bools[i]:
                labels[i] = target_label
        if prlf_len > 0:
            labels = proliferate_label(labels, target_label, prlf_len)
        return labels

    def proliferate_label(labels, prlf_target, prlf_len):
        N = len(labels)
        to_prlf = [False] * N
        for i in range(N):
            if labels[i] == prlf_target:
                for ii in range(i - prlf_len, i + prlf_len + 1):
                    if 0 <= ii <= N - 1:
                        to_prlf[ii] = True
        labels = np.array(labels)
        labels[to_prlf] = prlf_target
        return list(labels)
        return labels

    def calculate_euclidean_dist(arr):
        return np.sqrt(np.sum((arr ** 2), axis=1))

    def generate_outlier_mark(arr, label, outlier_label=' ',num_std=1.5):
        not_outlier_arr=[]
        for i in range(len(arr)):
            if label[i]!=outlier_label:
                not_outlier_arr.append(arr[i])
        not_outlier_arr=np.array(not_outlier_arr)
        m, std = np.mean(not_outlier_arr), np.std(not_outlier_arr)
        marks_keep = abs(arr - m) < num_std * std
        for i in range(len(arr)):
            if label[i]==outlier_label:
                marks_keep[i]=False
        return marks_keep

    std=[1.5,1.5,1.5,1.5]
    for i in range(len(std)):
        pupil_velocity_abs = calculate_euclidean_dist(pupil_velocity)
        is_not_jumping = generate_outlier_mark(pupil_velocity_abs,label=event_labels,outlier_label='saccade', num_std=std[i])
        is_jumping = ~is_not_jumping
        event_labels = mark_true_as(event_labels, is_jumping, 'saccade', prlf_len=1)  # safe margin of 2 frames (32ms)

    # 2. mark blink by blink_bools
    is_blinking = ~blink_bools
    event_labels = mark_true_as(event_labels, is_blinking, 'blink', prlf_len=1)  # also add a 2 frame margin

    for i in range(len(event_labels)):
        if event_labels[i] !='saccade' and event_labels[i] != 'blink':
            event_labels[i]='fixation'
    return event_labels

def device_level_accuracy(pred_one_condition, ground_truth_one_condition, screen_size, margin_scale=0.05):
    in_screen_net = []
    for i in range(len(pred_one_condition)):
        if ground_truth_one_condition[i][0] == -1 and ground_truth_one_condition[i][1] == -1:
            if ((0 - margin_scale * screen_size[0] <= pred_one_condition[i][0] <= (1 + margin_scale) * screen_size[0]) and
                    (0 - margin_scale * screen_size[1] <= pred_one_condition[i][1] <= (1 + margin_scale) * screen_size[1])):
                in_screen_net.append(0)
            else:
                in_screen_net.append(1)
        else:
            if ((0 - margin_scale * screen_size[0] <= pred_one_condition[i][0] <= (1 + margin_scale) * screen_size[0]) and
                    (0 - margin_scale * screen_size[1] <= pred_one_condition[i][1] <= (1 + margin_scale) * screen_size[1])):
                in_screen_net.append(1)
            else:
                in_screen_net.append(0)
    total_accurate = 0
    for i in range(len(pred_one_condition)):
        total_accurate = total_accurate + in_screen_net[i]
    accuracy = total_accurate / len(pred_one_condition)
    return accuracy

if __name__ == "__main__":
    import cv2
    import numpy as np
    import warp_norm
    from pt_module import StNet, StRefine
    import pandas as pd
    import copy
    from pt_module import StNet, StRefine
    from ipdb import set_trace as st
    import data_loader
    import trainer
    import argparse
    from config import get_config

    #模拟视线
    # points_array=[]
    # # 回调函数：鼠标点击输出点击的坐标
    # # （事件（鼠标移动、左键、右键），横坐标，纵坐标，组合键，setMouseCallback的userdata用于传参）
    # def mouse_callback(event, x, y, flags, userdata):
    #     # 如果鼠标左键点击，则输出横坐标和纵坐标
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print(f'({x}, {y})')
    #         points_array.append([x,y])
    #         # 在图像上绘制点
    #         cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    #
    #         # 在图像上添加坐标文本
    #         # （图像，文本内容，坐标点，字体类型，字体大小，颜色，字体粗细）
    #         cv2.putText(img, f'({x},{y})', (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    #         # 获取指定像素点的颜色
    #         # pixel_color = img[x, y]
    #         # print("颜色值BGR：", pixel_color)
    #
    #
    # img = cv2.imread('./testpart/5.jpg')
    #
    # # 创建窗口
    # cv2.namedWindow('Point Coordinates')
    #
    # # 将回调函数绑定到窗口
    # cv2.setMouseCallback('Point Coordinates', mouse_callback)
    #
    # # 显示图像
    # while True:
    #     cv2.imshow('Point Coordinates', img)
    #     k = cv2.waitKey(1) & 0xFF
    #     # 按esc键退出
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()

    # # 生成100个均匀分布的测试点
    # num_points = 100
    # canvas_size = [1600, 825]
    # # 生成均匀分布的x和y坐标
    # x_coordinates = np.random.uniform(0, canvas_size[0], num_points)
    # y_coordinates = np.random.uniform(0, canvas_size[1], num_points)
    # # 生成正态分布的x和y坐标
    # # mean = canvas_size[0] / 2  # 均值
    # # std_dev = canvas_size[0] / 6  # 标准差
    # # x_coordinates = np.random.normal(mean, std_dev, num_points)
    # # y_coordinates = np.random.normal(canvas_size[1] / 2, std_dev, num_points)
    # # x_coordinates = np.clip(x_coordinates, 0, canvas_size[0])
    # # y_coordinates = np.clip(y_coordinates, 0, canvas_size[1])
    # points_array = np.column_stack((x_coordinates, y_coordinates))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    pixel_scale_tan = np.array([0.202, 0.224])
    pixel_scale_chen = np.array([0.22, 0.235])


    """filter"""
    is_filtered = True
    if is_filtered:
        with open('./gaze_pred_smooth.pkl', 'rb') as fo:
            tinydict = pickle.load(fo, encoding='bytes')  #
    # else:
    #     with open('./gaze_pred.pkl', 'rb') as fo:
    #         tinydict = pickle.load(fo, encoding='bytes')  # que
    file_names = tinydict['file_name']
    filename = ['output1', 'output2', 'output3', 'output4', 'output5', 'output6', 'output7', 'output8']
    file_dict = []
    for file_name in file_names:
        file_name = str(file_name).strip('()').split(',')
        # print(file_name)
        for file in file_name:
            if (file != ''):
                print(file.strip("''").split('/')[6].split('.')[0][66:])
                if is_filtered:
                    file_dict.append(int(file.strip("''").split('/')[6].split('.')[0][66:]))
                else:
                    file_dict.append(int(file.strip("''").split('/')[4].split('.')[0][52:]))
    # print(file_dict)
    # print(len(file_dict))
    if is_filtered:
        with open('D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/dataset_dict_smooth.pkl', 'rb') as fo:
            tinydict2 = pickle.load(fo, encoding='bytes')  # que
    else:
        with open('D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/dataset_dict.pkl', 'rb') as fo:
            tinydict2 = pickle.load(fo, encoding='bytes')  # que
    ground_truth = [[]]
    pred = [[]]
    RMat = [[]]
    Ear =[[]]
    face_area=[[]]
    for i in range(len(file_dict)):
        number = 0
        ground_truth[number].append(tinydict['label'][i])
        pred[number].append(tinydict['pred_gaze'][i])
        RMat[number].append(tinydict2[i]['R'])
        Ear[number].append(tinydict2[i]['Ear'])
        face_area[number].append(tinydict2[i]['face_area_label'])
    # for i in range(46):
    #     print(ground_truth[i])
    for i in range(len(ground_truth)):
        ground_truth[i] = np.vstack(ground_truth[i])
        pred[i] = np.vstack(pred[i])
    # print(RMat[0][0])

    # 将pitchyaw转换成vector
    for i in range(len(ground_truth)):
        pred[i] = warp_norm.pitchyaw_to_vector(pred[i])

    # print(pred[0][0])

    # 将归一化向量还原
    org_pred = [[]]
    for i in range(len(ground_truth)):
        for j in range(len(RMat[i])):
            # print(pred[i][j])
            # print(RMat[i][j])
            if((RMat[i][j] == 0).all()):
                org_pred[i].append(pred[i][j])
            else:
                org_pred[i].append(np.dot(np.linalg.inv(RMat[i][j]), pred[i][j].T))
                # print(org_pred[i][j])
    # print(org_pred[0][0])

    pred_gc = [[]]
    for i in range(len(ground_truth)):
        for j in range(len(org_pred[i])):
            pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale_tan))

    # print(pred_gc[0])
    org_tan = np.array([800, 0])  # tan 1600*825
    org_chen = np.array([650, 0])  # tan 1300*720
    pred_gc_org = [[]]
    for i in range(len(ground_truth)):
        pred_gc_org[i] = org_chen + pred_gc[i]
    # print(len(pred_gc))
    # points_array=pred_gc[0]


    num_video=0
    # for num_video in range(len(filename)):
    points_array=np.array(pred_gc_org[num_video])
    # points_array_pt=np.array(final_pred[num_video])
    # print(len(Ear))
    points_array=np.array(points_array)
    # print(points_array.shape)
    # print(points_array)
    blink=np.full((len(points_array), 1), True)
    for i in range(len(Ear[0])):
        if Ear[0][i] < 0:
            # print(i+1)
            blink[i]=False
    # print(blink.shape)
    event_label=mark_saccade_and_blink(points_array,blink)
    # print(event_label)
    num_fix=0
    for i in range(len(points_array)-1):
        if event_label[i] == 'saccade' and event_label[i+1] == 'fixation':
            num_fix = num_fix + 1

    # print(num_fix)
    history=[[], [], [], [], []]
    for i in range(int(len(points_array))):
        if(event_label[i] == 'fixation'):
            history[int(face_area[0][i])].append(points_array[i])
    numf=0
    for i in range(5):
        numf = 0
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot()
        rect = plt.Rectangle((0, 0), 1600, 825, edgecolor='r', facecolor='None')
        ax.add_patch(rect)
        for j in range(int(len(history[i]))):
            plt.scatter(history[i][j][0], history[i][j][1], marker='o', color='r', label=f'fixation')
            numf = numf + 1
            # if i>0 and event_label[i] == 'fixation' and event_label[i-1] == 'fixation':
            #     plt.arrow(points_array[i-1][0], points_array[i-1][1], points_array[i][0] - points_array[i-1][0], points_array[i][1] - points_array[i-1][1], color='b',alpha=0.5)
            if j == 0:
                plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.show()
        dst_filename = 'debug/face_center_in_area_' + str(i) + '_history_x_vs_y_xgaze' + '.jpg'
        plt.title('face_center_in_area_' + str(i) + '_history_x_vs_y_xgaze' + '.jpg')
        plt.savefig(dst_filename)
        plt.clf()
        print(numf)


    #classifiction plt

    # for i in range(int(len(points_array)/20)):
    #     if event_label[i] == 'fixation':
    #         plt.scatter(points_array[i][0], points_array[i][1], marker='o', color='r', label=f'fixation')
    #     elif event_label[i] == 'saccade':
    #         plt.scatter(points_array[i][0], points_array[i][1], marker='o', color='g', label=f'saccade')
    #     elif event_label[i] == 'blink':
    #         plt.scatter(points_array[i][0], points_array[i][1], marker='o', color='y', label=f'blink')
    #     if i>0:
    #         plt.arrow(points_array[i-1][0], points_array[i-1][1],
    #                   points_array[i][0] - points_array[i-1][0],
    #                   points_array[i][1] - points_array[i-1][1], color='y', alpha=0.5)
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # # plt.show()
    # dst_filename = 'debug/' + filename[num_video] + '_y_vs_x_history' + '.jpg'
    # plt.title(filename[num_video] + '_y_vs_x_history')
    # plt.savefig(dst_filename)
    # plt.clf()
    #
    #
    # for i in range(int(len(points_array)/20)):
    #     if event_label[i] == 'fixation':
    #         plt.scatter(i, points_array[i][0], marker='o', color='r', label=f'fixation')
    #     elif event_label[i] == 'saccade':
    #         plt.scatter(i, points_array[i][0], marker='o', color='g', label=f'saccade')
    #     elif event_label[i] == 'blink':
    #         plt.scatter(i, points_array[i][0], marker='o', color='y', label=f'blink')
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.xlabel('frames')
    # plt.ylabel('X')
    # plt.ylim((-3000, 5000))
    # # plt.show()
    # dst_filename = 'debug/' + filename[num_video] + '_x_vs_time_xgaze' + '.jpg'
    # plt.title(filename[num_video] + '_x_vs_time_xgaze')
    # plt.savefig(dst_filename)
    # plt.clf()
    #
    # for i in range(int(len(points_array)/20)):
    #     if event_label[i] == 'fixation':
    #         plt.scatter(i, points_array[i][1], marker='o', color='r', label=f'fixation')
    #     elif event_label[i] == 'saccade':
    #         plt.scatter(i, points_array[i][1], marker='o', color='g', label=f'saccade')
    #     elif event_label[i] == 'blink':
    #         plt.scatter(i, points_array[i][1], marker='o', color='y', label=f'blink')
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.xlabel('frames')
    # plt.ylabel('Y')
    # plt.ylim((-3000, 5000))
    # # plt.show()
    # dst_filename='debug/'+ filename[num_video] +'_y_vs_time_xgaze' + '.jpg'
    # plt.title(filename[num_video] + '_y_vs_time_xgaze')
    # plt.savefig(dst_filename)
    # plt.clf()
    #
    # for i in range(int(len(points_array)/20)):
    #     if event_label[i] == 'fixation':
    #         plt.scatter(i, points_array[i][0], marker='o', color='r', label=f'fixation')
    #     elif event_label[i] == 'saccade':
    #         plt.scatter(i, points_array[i][0], marker='o', color='g', label=f'saccade')
    #     elif event_label[i] == 'blink':
    #         plt.scatter(i, points_array[i][0], marker='o', color='y', label=f'blink')
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.xlabel('frames')
    # plt.ylabel('X')
    # plt.ylim((-3000, 5000))
    # # plt.show()
    # dst_filename = 'debug/' + filename[num_video] + '_x_vs_time_pt' + '.jpg'
    # plt.title(filename[num_video] + '_x_vs_time_pt')
    # plt.savefig(dst_filename)
    # plt.clf()
    #
    # for i in range(int(len(points_array)/20)):
    #     if event_label[i] == 'fixation':
    #         plt.scatter(i, points_array[i][1], marker='o', color='r', label=f'fixation')
    #     elif event_label[i] == 'saccade':
    #         plt.scatter(i, points_array[i][1], marker='o', color='g', label=f'saccade')
    #     elif event_label[i] == 'blink':
    #         plt.scatter(i, points_array[i][1], marker='o', color='y', label=f'blink')
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # plt.xlabel('frames')
    # plt.ylabel('X')
    # plt.ylim((-3000, 5000))
    # # plt.show()
    # dst_filename = 'debug/' + filename[num_video] + '_y_vs_time_pt' + '.jpg'
    # plt.title(filename[num_video] + '_y_vs_time_pt')
    # plt.savefig(dst_filename)
    # plt.clf()
    #todo: 1.fixation 100ms~500ms 做5次筛选  2.range->(mean-80,mean+80)   3*.filter 5frames hr ht


    #todo:1.not upright condition 分区    2. 4 condition xy准确率/设备级准确率

    # predict test gaze


    config, unparsed = get_config()
    config.is_train = False
    config.batch_size = 10
    config.use_gpu = True
    config.ckpt_dir = './ckpt'
    config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    test_data = data_loader.get_test_loader('D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/preprocessed_labels.csv',
                                            batch_size=config.batch_size,
                                            data_path='D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/preprocessed_images',
                                            num_workers=0)
    xgaze = trainer.Trainer(config, test_data)
    xgaze.test(save_fname_pkl='gaze_pred_test.pkl', save_fname_txt='results_test.txt')

    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    dataset_pkl_dir = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/dataset_dict.pkl'
    save_fname_pkl = 'gaze_pred_test.pkl'
    save_fname_txt = 'results_test.txt'
    model_dir = 'D:/Deeplearning/GazeNormalization-main/models'
    state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
    state_path = model_dir + '/' + state_name
    condition_label = ['正常灯光 && upright', '正常灯光 && not upright',
                       '正常灯光 && upright && 跨设备', '正常灯光 && not upright && 跨设备',
                       '仅开台灯 && upright', '仅开台灯 && not upright',
                       '仅开台灯 && upright && 跨设备', '仅开台灯 && not upright && 跨设备',
                       ]

    with open('./'+save_fname_pkl, 'rb') as fo:
        tinydict_test = pickle.load(fo, encoding='bytes')  # que
    file_names_test = tinydict_test['file_name']
    file_dict_test = []
    for file_name in file_names_test:
        file_name = str(file_name).strip('()').split(',')
        # print(file_name)
        for file in file_name:
            if (file != ''):
                # print(file.strip("''").split('/')[5].split('.')[0][40:])
                file_dict_test.append(int(file.strip("''").split('/')[5].split('.')[0][40:]))
    # print(file_dict_test)
    # print(len(file_dict_test))

    with open(dataset_pkl_dir, 'rb') as fo:
        tinydict2_test = pickle.load(fo, encoding='bytes')  # que


    def get_condition_number(file_dict):
        if (1 <= file_dict <= 50):
            return 0
        elif (51 <= file_dict <= 100):
            return 1
        elif (101 <= file_dict <= 140):
            return 2
        elif (141 <= file_dict <= 180):
            return 3
        elif (181 <= file_dict <= 230):
            return 4
        elif (231 <= file_dict <= 280):
            return 5
        elif (281 <= file_dict <= 320):
            return 6
        elif (321 <= file_dict <= 360):
            return 7


    ground_truth_test = [[], [], [], [], [], [], [], []]
    pred_test = [[], [], [], [], [], [], [], []]
    RMat_test = [[], [], [], [], [], [], [], []]
    Ear_test = [[], [], [], [], [], [], [], []]
    face_area_test = [[], [], [], [], [], [], [], []]
    for i in range(len(file_dict_test)):
        number = get_condition_number(file_dict_test[i])
        if number >= 0:
            ground_truth_test[number].append(tinydict_test['label'][i])
            pred_test[number].append(tinydict_test['pred_gaze'][i])
            print(i)
            print(len(file_dict_test))
            RMat_test[number].append(tinydict2_test[i]['R'])
            # Ear_test[number].append(tinydict2_test[i]['Ear'])
            face_area_test[number].append(tinydict2_test[i]['face_area_label'])

    for i in range(len(ground_truth_test)):
        ground_truth_test[i] = np.vstack(ground_truth_test[i])
        pred_test[i] = np.vstack(pred_test[i])
    # 将pitchyaw转换成vector
    for i in range(len(ground_truth_test)):
        pred_test[i] = warp_norm.pitchyaw_to_vector(pred_test[i])

    org_pred_test = [[], [], [], [], [], [], [], []]
    for i in range(len(ground_truth_test)):
        for j in range(len(RMat_test[i])):
            org_pred_test[i].append(np.dot(np.linalg.inv(RMat_test[i][j]), pred_test[i][j].T))

    pixel_scale_tan = np.array([0.202, 0.224])
    pred_gc_xgaze_original_test = [[], [], [], [], [], [], [], []]
    pred_gc_test = [[], [], [], [], [], [], [], []]
    for i in range(len(ground_truth_test)):
        for j in range(len(pred_test[i])):
            if i >= 0:
                org_pred_test_copy = copy.deepcopy(org_pred_test)
                pred_gc_test[i].append(warp_norm.vector_to_gc(org_pred_test_copy[i][j], pixel_scale_chen,face_center=-600*0.6))######
                org_pred_test_copy = copy.deepcopy(org_pred_test)
                pred_gc_xgaze_original_test[i].append(warp_norm.vector_to_gc(org_pred_test_copy[i][j], pixel_scale_chen,face_center=-600))######
            # else:
            #     pred_gc_test[i].append(warp_norm.vector_to_gc(org_pred_test[i][j], pixel_scale_chen))
    org_tan = np.array([800, 0])  # tan 1600*825
    org_chen = np.array([650, 0])  # chen 1300*720
    pred_gc_org_xgaze_original_test = [[], [], [], [], [], [], [], []]
    pred_gc_org_test = [[], [], [], [], [], [], [], []]
    for i in range(len(ground_truth_test)):
        if i >= 0:
            # pred_gc_org_test[i] = org_tan + pred_gc_test[i]
            # pred_gc_org_xgaze_original_test[i] = org_tan + pred_gc_xgaze_original_test[i]
        # else:
            pred_gc_org_test[i] = org_chen + pred_gc_test[i]
            pred_gc_org_xgaze_original_test[i] = org_chen + pred_gc_xgaze_original_test[i]
    Initnet_xerrors = []
    Initnet_yerrors = []
    print(len(ground_truth_test))
    for i in range(len(ground_truth_test)):
        total_xerrors = 0
        total_yerrors = 0
        for j in range(len(ground_truth_test[i])):
            total_xerrors = total_xerrors + abs(pred_gc_org_xgaze_original_test[i][j][0] - ground_truth_test[i][j][0])
            total_yerrors = total_yerrors + abs(pred_gc_org_xgaze_original_test[i][j][1] - ground_truth_test[i][j][1])
        Initnet_xerrors.append(total_xerrors / (len(ground_truth_test[i])))
        Initnet_yerrors.append(total_yerrors / (len(ground_truth_test[i])))
    Initnet_xerrors_cm = []
    Initnet_yerrors_cm = []
    for i in range(len(ground_truth_test)):
        if i >= 0:
            # Initnet_xerrors_cm.append(Initnet_xerrors[i] * 0.1 * pixel_scale_tan[0])
            # Initnet_yerrors_cm.append(Initnet_yerrors[i] * 0.1 * pixel_scale_tan[1])
        # else:
            Initnet_xerrors_cm.append(Initnet_xerrors[i] * 0.1 * pixel_scale_chen[0])
            Initnet_yerrors_cm.append(Initnet_yerrors[i] * 0.1 * pixel_scale_chen[1])
    print('Initnet x&y errors:')
    print(Initnet_xerrors_cm)
    print(Initnet_yerrors_cm)


    # device_level_accuracy
    Initnet_device_accuracy = []
    for i in range(len(ground_truth_test)):
        # accuracy = device_level_accuracy(pred_gc_org_test[i], ground_truth_test[i], (1600, 825), margin_scale=0.05)
        accuracy = device_level_accuracy(pred_gc_org_test[i], ground_truth_test[i], (1300, 720), margin_scale=0.05)
        Initnet_device_accuracy.append(accuracy)
    print('Initnet_device_accuracy:')
    print(Initnet_device_accuracy)

    # Zoom Module
    """using history range and screen size to zoom pred range"""
    zoom_history_range = [[], [], [], [], []]
    center = [[], [], [], [], []]
    # screen_size = np.array([1600, 825])
    screen_size = np.array([1300, 720])
    range_scale = 0.9
    zoom_scale = []
    for i in range(len(history)):
        if(len(history[i]) == 0):
            zoom_scale.append(np.array([1, 1]))
            continue
        history_percentile = np.percentile(history[i],
                                        [100 * (1 - range_scale) / 2, 100 - 100 * (1 - range_scale) / 2],
                                        axis=0)  # get percentile
        # print(pred_percentile)
        history_percentile_range = history_percentile[1] - history_percentile[0]
        # print(pred_percentile_range)
        truth_percentile_range = screen_size * range_scale
        zoom_scale.append(truth_percentile_range / history_percentile_range)
    print(zoom_scale)

    pred_gc_zoomed = [[], [], [], [], [], [], [], []]
    for i in range(len(history)):
        total_history = np.array([0, 0])
        for j in range(int(len(history[i]))):
            total_history = total_history + history[i][j]
        center[i] = total_history / int(len(history[i]))
    for i in range(len(pred_gc_org_xgaze_original_test)):
        for j in range(len(pred_gc_org_xgaze_original_test[i])):
            if((zoom_scale[face_area_test[i][j]] == [1, 1]).any()):
                pred_gc_zoomed[i].append(pred_gc_org_xgaze_original_test[i][j])
            else:
                pred_gc_zoomed[i].append((pred_gc_org_xgaze_original_test[i][j] - center[face_area_test[i][j]]) * zoom_scale[face_area_test[i][j]] + center[face_area_test[i][j]])
    Zoomed_pred_copy = copy.deepcopy(pred_gc_zoomed)  # deep copy

    Zoom_xerrors = []
    Zoom_yerrors = []
    for i in range(len(ground_truth_test)):
        total_xerrors = 0
        total_yerrors = 0
        for j in range(len(ground_truth_test[i])):
            total_xerrors = total_xerrors + abs(pred_gc_zoomed[i][j][0] - ground_truth_test[i][j][0])
            total_yerrors = total_yerrors + abs(pred_gc_zoomed[i][j][1] - ground_truth_test[i][j][1])
        Zoom_xerrors.append(total_xerrors / (len(ground_truth_test[i])))
        Zoom_yerrors.append(total_yerrors / (len(ground_truth_test[i])))

    Zoom_xerrors_cm = []
    Zoom_yerrors_cm = []
    for i in range(len(ground_truth_test)):
        if i >= 0:
            Zoom_xerrors_cm.append(Zoom_xerrors[i] * 0.1 * pixel_scale_tan[0])
            Zoom_yerrors_cm.append(Zoom_yerrors[i] * 0.1 * pixel_scale_tan[1])
        # else:
        #     SC_xerrors_cm.append(SC_xerrors[i] * 0.1 * pixel_scale_chen[0])
        #     SC_yerrors_cm.append(SC_yerrors[i] * 0.1 * pixel_scale_chen[1])
    print('Zoom_ x&y errors:')
    print(Zoom_xerrors_cm)
    print(Zoom_yerrors_cm)

    # device_level_accuracy
    Zoom_device_accuracy = []
    for i in range(len(ground_truth_test)):
        # accuracy = device_level_accuracy(pred_gc_zoomed[i], ground_truth_test[i], (1600, 825), margin_scale=0.05)
        accuracy = device_level_accuracy(pred_gc_zoomed[i], ground_truth_test[i], (1300, 720), margin_scale=0.05)

        Zoom_device_accuracy.append(accuracy)
    print('Zoom_device_accuracy:')
    print(Zoom_device_accuracy)

    # SC Module
    gtr = [[], [], [], [], []]
    SC_history_average = [[], [], [], [], []]
    offset = [[], [], [], [], []]
    for i in range(len(history)):
        total_history = [0, 0]
        if i >= 0:
            # gtr[i] = [800, 412.5]
        # else:
            gtr[i] = [650, 360]
        if(len(history[i]) == 0):
            offset[i] = [0, 0]
            continue
        for j in range(len(history[i])):
            total_history = total_history + history[i][j]
        SC_history_average[i] = total_history / int(len(history[i]))
        offset[i] = SC_history_average[i] - gtr[i]

    refine_pred = [[], [], [], [], [], [], [], []]
    for i in range(len(ground_truth_test)):
        for j in range(len(ground_truth_test[i])):
            refine_pred[i].append(pred_gc_zoomed[i][j] - offset[face_area_test[i][j]])

    refine_pred_copy = copy.deepcopy(refine_pred)#deep copy

    SC_xerrors = []
    SC_yerrors = []
    for i in range(len(ground_truth_test)):
        total_xerrors = 0
        total_yerrors = 0
        for j in range(len(ground_truth_test[i])):
            total_xerrors = total_xerrors + abs(refine_pred[i][j][0] - ground_truth_test[i][j][0])
            total_yerrors = total_yerrors + abs(refine_pred[i][j][1] - ground_truth_test[i][j][1])
        SC_xerrors.append(total_xerrors / (len(ground_truth_test[i])))
        SC_yerrors.append(total_yerrors / (len(ground_truth_test[i])))

    SC_xerrors_cm = []
    SC_yerrors_cm = []
    for i in range(len(ground_truth_test)):
        if i >= 0:
            SC_xerrors_cm.append(SC_xerrors[i] * 0.1 * pixel_scale_tan[0])
            SC_yerrors_cm.append(SC_yerrors[i] * 0.1 * pixel_scale_tan[1])
        # else:
        #     SC_xerrors_cm.append(SC_xerrors[i] * 0.1 * pixel_scale_chen[0])
        #     SC_yerrors_cm.append(SC_yerrors[i] * 0.1 * pixel_scale_chen[1])
    print('SC x&y errors:')
    print(SC_xerrors_cm)
    print(SC_yerrors_cm)

    # device_level_accuracy
    SC_device_accuracy = []
    for i in range(len(ground_truth_test)):
        # accuracy = device_level_accuracy(refine_pred[i], ground_truth_test[i], (1600, 825), margin_scale=0.05)
        accuracy = device_level_accuracy(refine_pred[i], ground_truth_test[i], (1300, 720), margin_scale=0.05)
        SC_device_accuracy.append(accuracy)
    print('SC_device_accuracy:')
    print(SC_device_accuracy)



    # plot
    for i in range(len(ground_truth_test)):
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot()
        if i >= 0:
            rect = plt.Rectangle((0, 0), 1300, 720, edgecolor='r', facecolor='None')
        # else:
        #     rect = plt.Rectangle((0, 0), 1300, 720, edgecolor='r', facecolor='None')
        ax.add_patch(rect)
        for j in range(len(pred_test[i])):
            plt.scatter(pred_gc_org_xgaze_original_test[i][j][0], pred_gc_org_xgaze_original_test[i][j][1], marker='o', color=colors[3],
                        label=f'Xgaze Pred')
            plt.scatter(Zoomed_pred_copy[i][j][0], Zoomed_pred_copy[i][j][1], marker='o', color=colors[2],
                        label=f'zoomed Pred')
            plt.scatter(refine_pred_copy[i][j][0], refine_pred_copy[i][j][1], marker='o', color=colors[0],
                        label=f'SC Pred')
            plt.scatter(ground_truth_test[i][j][0], ground_truth_test[i][j][1], marker='x', color=colors[1],
                        label=f'True')

            plt.arrow(ground_truth_test[i][j][0], ground_truth_test[i][j][1],
                      pred_gc_org_xgaze_original_test[i][j][0] - ground_truth_test[i][j][0],
                      pred_gc_org_xgaze_original_test[i][j][1] - ground_truth_test[i][j][1], color=colors[3], alpha=0.5)
            plt.arrow(ground_truth_test[i][j][0], ground_truth_test[i][j][1],
                      refine_pred_copy[i][j][0] - ground_truth_test[i][j][0],
                      refine_pred_copy[i][j][1] - ground_truth_test[i][j][1], color=colors[0], alpha=0.5)
            plt.arrow(ground_truth_test[i][j][0], ground_truth_test[i][j][1],
                      Zoomed_pred_copy[i][j][0] - ground_truth_test[i][j][0],
                      Zoomed_pred_copy[i][j][1] - ground_truth_test[i][j][1], color=colors[2], alpha=0.5)
            if (j == 0):
                plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.legend()
        plt.tight_layout()
        # plt.show()
        dst_filename = 'debug/test_' + condition_label[i] + '_x_vs_y_PT' + '.jpg'
        plt.title('test_' + condition_label[i] + '_x_vs_y_PT')
        plt.savefig(dst_filename)
        plt.clf()
#todo:在所有condition中误差、跨设备、所有fixation做history(去除检测的图片)    show:实时显示跨设备的标注在后50%的视频中    ddl：周六晚上八点