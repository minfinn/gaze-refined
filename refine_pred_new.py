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
import pickle
import pandas as pd
import copy
from pt_module import StNet,StRefine
from ipdb import set_trace as st
import refine


colors = plt.cm.viridis(np.linspace(0, 1, 4))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

model_dir = 'D:/Deeplearning/GazeNormalization-main/models'
state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
state_path = model_dir + '/' + state_name
num_history=150
condition_label=[r'glass && upright tan', r'glass && upright chen',
                 r'no glass && upright tan', r'no glass && upright chen',
                 r'glass && not upright tan', r'glass && not upright chen',
                 r'no glass && not upright tan', r'no glass && not upright chen',
                 r'glass && 白天室内光照 tan', r'glass && 白天室内光照 chen',
                 r'no glass && 白天室内光照 tan', r'no glass && 白天室内光照 chen',
                 r'glass && 黑暗环境仅开台灯 tan', r'glass && 黑暗环境仅开台灯 chen',
                 r'no glass && 黑暗环境仅开台灯 tan', r'no glass && 黑暗环境仅开台灯 chen',
                 r'glass && 黑暗环境外部灯光 tan', r'glass && 黑暗环境外部灯光 chen',
                 r'no glass && 黑暗环境外部灯光 tan', r'no glass && 黑暗环境外部灯光 chen',
                 r'晚上室内正常光照 && glass && no mask tan', r'晚上室内正常光照 && glass && no mask chen',
                 r'晚上室内正常光照 && glass && 设备距离大（70cm+） tan', r'晚上室内正常光照 && glass && 设备距离大（70cm+） chen',
                 r'晚上室内正常光照 && glass && 设备距离中（45-48cm） tan', r'晚上室内正常光照 && glass && 设备距离中（45-48cm） chen',
                 r'晚上室内正常光照 && glass && 设备距离小（32-35cm） tan', r'晚上室内正常光照 && glass && 设备距离小（32-35cm） chen',
                 r'晚上室内正常光照 && glass && 设备倾斜角大（45°） tan', r'晚上室内正常光照 && glass && 设备倾斜角大（45°） chen',
                 r'晚上室内正常光照 && glass && 设备倾斜角中（30°） tan', r'晚上室内正常光照 && glass && 设备倾斜角中（30°） chen',
                 r'晚上室内正常光照 && glass && 设备倾斜角小（15°） tan', r'晚上室内正常光照 && glass && 设备倾斜角小（15°） chen',
                 r'正常距离 看电脑 tan',
                 r'正常距离 看电脑 chen',
                 r'正常距离 看手机 tan',
                 r'正常距离 看手机 chen',
                 r'极端距离 看电脑 tan',
                 r'极端距离 看电脑 chen',
                 r'极端距离 看手机 tan',
                 r'极端距离 看手机 chen',
                 r'手机平放 看电脑 tan',
                 r'手机平放 看电脑 chen',
                 r'手机平放 看手机 tan',
                 r'手机平放 看手机 chen',
                 r'设备距离中', r'设备距离小', r'设备距离大', r'设备倾斜角中', r'设备倾斜角小', r'设备倾斜角大']
# print(len(condition_label))

pixel_scale_tan = np.array([0.202, 0.224])
pixel_scale_chen = np.array([0.22, 0.235])

with open('./gaze_pred_old.pkl', 'rb') as fo:
    tinydict = pickle.load(fo, encoding='bytes')#que
file_names = tinydict['file_name']
file_dict = []
for file_name in file_names:
    file_name = str(file_name).strip('()').split(',')
    # print(file_name)
    for file in file_name:
        if(file!=''):
            # print(file.strip("''").split('/')[4].split('.')[0][40:])
            # print(file.strip("''").split('/')[4].split('.')[0][40:])
            file_dict.append(int(file.strip("''").split('/')[4].split('.')[0][40:]))
# print(file_dict)
print(len(file_dict))

with open('D:/Deeplearning/dataset/Dataset_com/dataset_dict.pkl', 'rb') as fo:
    tinydict2 = pickle.load(fo, encoding='bytes')#que

def get_condition_number(file_dict):
    if (1 <= file_dict <= 100):
        return 0
    elif (101 <= file_dict <= 200):
        return 1
    elif (201 <= file_dict <= 300):
        return 2
    elif (301 <= file_dict <= 400):
        return 3
    elif (401 <= file_dict <= 500):
        return 4
    elif (501 <= file_dict <= 600):
        return 5
    elif (601 <= file_dict <= 700):
        return 6
    elif (701 <= file_dict <= 800):
        return 7
    elif (801 <= file_dict <= 850):
        return 8
    elif (851 <= file_dict <= 900):
        return 9
    elif (901 <= file_dict <= 950):
        return 10
    elif (951 <= file_dict <= 1000):
        return 11
    elif (1001 <= file_dict <= 1050):
        return 12
    elif (1051 <= file_dict <= 1100):
        return 13
    elif (1101 <= file_dict <= 1150):
        return 14
    elif (1151 <= file_dict <= 1200):
        return 15
    elif (1201 <= file_dict <= 1250):
        return 16
    elif (1251 <= file_dict <= 1300):
        return 17
    elif (1301 <= file_dict <= 1350):
        return 18
    elif (1351 <= file_dict <= 1400):
        return 19
    elif (1401 <= file_dict <= 1440):
        return -1
    elif (1441 <= file_dict <= 1520):
        return 20
    elif (1521 <= file_dict <= 1600):
        return 21
    elif (1601 <= file_dict <= 1620):
        return 22
    elif (1621 <= file_dict <= 1640):
        return 23
    elif (1641 <= file_dict <= 1700):
        return 24
    elif (1701 <= file_dict <= 1760):
        return 25
    elif (1761 <= file_dict <= 1780):
        return 26
    elif (1781 <= file_dict <= 1800):
        return 27
    elif (1801 <= file_dict <= 1820):
        return 28
    elif (1821 <= file_dict <= 1840):
        return 29
    elif (1841 <= file_dict <= 1900):
        return 30
    elif (1901 <= file_dict <= 1960):
        return 31
    elif (1961 <= file_dict <= 1980):
        return 32
    elif (1981 <= file_dict <= 2000):
        return 33
    elif (2001 <= file_dict <= 2050):
        return 34
    elif (2051 <= file_dict <= 2100):
        return 35
    elif (2101 <= file_dict <= 2150):
        return 36
    elif (2151 <= file_dict <= 2200):
        return 37
    elif (2201 <= file_dict <= 2250):
        return 38
    elif (2251 <= file_dict <= 2300):
        return 39
    elif (2301 <= file_dict <= 2350):
        return 40
    elif (2351 <= file_dict <= 2400):
        return 41
    elif (2401 <= file_dict <= 2450):
        return 42
    elif (2451 <= file_dict <= 2500):
        return 43
    elif (2501 <= file_dict <= 2550):
        return 44
    elif (2551 <= file_dict <= 2600):
        return 45
    elif (2601 <= file_dict <= 2800):
        return 46
    elif (2801 <= file_dict <= 3000):
        return 47
    elif (3001 <= file_dict <= 3200):
        return 48
    elif (3201 <= file_dict <= 3400):
        return 49
    elif (3401 <= file_dict <= 3600):
        return 50
    elif (3601 <= file_dict <= 3800):
        return 51

ground_truth = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pred = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
RMat = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(file_dict)):
    number = get_condition_number(file_dict[i])
    if number >= 0:
        ground_truth[number].append(tinydict['label'][i])
        pred[number].append(tinydict['pred_gaze'][i])
        RMat[number].append(tinydict2[i]['R'])

for i in range(len(ground_truth)):
    ground_truth[i] = np.vstack(ground_truth[i])
    pred[i] = np.vstack(pred[i])

# 还原归一化
pred_gc_org = refine.Revert_normalization(pred,RMat)

pred_xerrors_cm, pred_yerrors_cm = refine.PoG_errors(pred_gc_org, ground_truth)

print('pred errors:')
print(pred_xerrors_cm)
print(pred_yerrors_cm)




device_level_accuracy = refine.in_screen_percentage(pred_gc_org)
print(device_level_accuracy)


grand_truth_history = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pred_history = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(len(pred_gc_org)):
    if i==36 or i==37 or i==40 or i==41 or i==44 or i==45:
        for j in range(int(len(pred_gc_org[i-2])/2)):
            pred_history[i].append(pred_gc_org[i-2][j])
            grand_truth_history[i].append(ground_truth[i-2][j])
    else:
        for j in range(int(len(pred_gc_org[i])/2)):
            pred_history[i].append(pred_gc_org[i][j])
            grand_truth_history[i].append(ground_truth[i][j])
# Zoom
pred_gc_zoomed = refine.Zoom(pred_history, grand_truth_history, pred_gc_org, range_scale=0.9)
Zoom_xerrors_cm, Zoom_yerrors_cm = refine.PoG_errors(pred_gc_zoomed,ground_truth)
print('Zoom errors:')
print(Zoom_xerrors_cm)
print(Zoom_yerrors_cm)

device_level_accuracy = refine.in_screen_percentage(pred_gc_zoomed)
print(device_level_accuracy)


#SC Module
grand_truth_history = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
pred_history = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
for i in range(len(pred_gc_zoomed)):
    if i == 36 or i == 37 or i == 40 or i == 41 or i == 44 or i == 45:
        for j in range(int(len(pred_gc_zoomed[i - 2]) / 2)):
            pred_history[i].append(pred_gc_zoomed[i - 2][j])
            grand_truth_history[i].append(ground_truth[i - 2][j])
    else:
        for j in range(int(len(pred_gc_zoomed[i]) / 2)):
            pred_history[i].append(pred_gc_zoomed[i][j])
            grand_truth_history[i].append(ground_truth[i][j])
refine_pred=refine.Self_Calibration(pred_history, grand_truth_history, pred_gc_zoomed)
# for plot
refine_pred_copy=copy.deepcopy(refine_pred)

SC_xerrors_cm, SC_yerrors_cm = refine.PoG_errors(refine_pred,ground_truth)
print('SC errors:')
print(SC_xerrors_cm)
print(SC_yerrors_cm)

device_level_accuracy = refine.in_screen_percentage(refine_pred)
print(device_level_accuracy)



#PT
# #history_heatmap
# def create_history_gaze_path_map(PoG_pxs, history_trajectory_map_size=(256, 144), actual_screen_size=(1920, 1080),
#                                  guassian_blur=(15, 15)):
#     # xys = sample['PoG_history_gt'][sample['PoG_history_gt_validity']]
#     xys = PoG_pxs
#     # history_trajectory_map_size = 256, 144
#     # actual_screen_size = 1920, 1080
#     w, h = 256, 144
#
#     trajmap = np.zeros((h, w))
#     xys_copy = copy.deepcopy(xys)
#
#     xys_copy[:, 0] *= (w / actual_screen_size[0])
#     xys_copy[:, 1] *= (h / actual_screen_size[1])
#     arrPt = np.array(xys_copy, np.int32).reshape((-1, 1, 2))
#
#     trajmap = cv2.polylines(trajmap, [arrPt], isClosed=False, color=(1.0,), thickness=2)
#     trajmap = cv2.GaussianBlur(trajmap, guassian_blur, 3)
#     if (w, h) != history_trajectory_map_size:
#         trajmap = cv2.resize(trajmap, history_trajectory_map_size)
#     trajmap = normalise_arr(trajmap)
#     plt.imshow(trajmap, origin='upper')
#     plt.show()
#     trajmap.shape
#     return trajmap
#
# def normalise_arr(arr):
#     mmax, mmin = np.max(arr), np.min(arr)
#     assert mmax > mmin
#     arr = (arr - mmin +1e-8)/(mmax - mmin + 2e-8)
#     return arr

# history=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
# for i in range(52):
#     for j in range(int(len(refine_pred[i])/2)):
#         history[i].append(np.array(refine_pred[i][j]))
    # print(np.array(history[i]).shape)
# create_history_gaze_path_map(np.array(history[0]),actual_screen_size=(1600, 825))

#PT Module
# st_refine_tan = StRefine(StNet_path=state_path, full_screen_size=(1600, 825))
# st_refine_chen = StRefine(StNet_path=state_path, full_screen_size=(1300, 720))
# result,valid=st_refine.refine(np.array(refine_pred[0][0]),history[0])
# print(refine_pred[0][0])
# print(result.detach().cpu().numpy())



# for i in range(8):
#     if(i%2==0):
#         output, valid = st_refine_tan.refine(np.array(refine_pred[i][0]), np.array(history[i]))
#     else:
#         output, valid = st_refine_chen.refine(np.array(refine_pred[i][0]), np.array(history[i]))

#
# size=[1200,1000]
# margin_size=0
#
#
# pred_rescaled=refine.rescale(refine_pred,size=size)
# pred_with_margin,size_with_margin = refine.margin(pred_rescaled,size=size,margin_size=margin_size)


# history=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
# for i in range(52):
#     for j in range(int(len(pred_with_margin[i])/2)):
#         history[i].append(np.array(pred_with_margin[i][j]))

# st_refine = StRefine(StNet_path=state_path, full_screen_size=size_with_margin)


# final_pred=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
# for i in range(52):
#     for j in range(len(refine_pred[i])):
#         show=False
#         output, valid = st_refine.refine(np.array(pred_with_margin[i][j]), np.array(history[i]),show=show)
#         # if(i==2 or i==3):
#         #     output, valid = st_refine.refine(np.array(refine_pred[i][j]), np.array(history[i-2]))
#         # else:
#         #     output, valid = st_refine.refine(np.array(refine_pred[i][j]), np.array(history[i]))
#         # print(refine_pred[i][j],output,valid)
#         # print(output.detach().cpu().numpy())
#         if valid:
#             final_pred[i].append(output.detach().cpu().numpy())
#         else:
#             final_pred[i].append(output)
#     # print(np.array(final_pred[i]).shape)
#
#
# final_pred_no_margin,size=remove_margin(final_pred,size_with_margin=size_with_margin,margin_size=margin_size)
# final_pred_rescaled=rev_rescale(final_pred_no_margin,pred_size=size)
# final_pred=final_pred_rescaled

# PT_xerrors_cm, PT_yerrors_cm = refine.PoG_errors(final_pred, ground_truth)
# print('PT errors:')
# print(PT_xerrors_cm)
# print(PT_yerrors_cm)


# device_level_accuracy = refine.in_screen_percentage(final_pred)
# print(device_level_accuracy)

for i in range(52):
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()
    if i%2==0 or i >= 46:
        rect = plt.Rectangle((0, 0), 1600, 825, edgecolor='r', facecolor='None')
    else:
        rect = plt.Rectangle((0, 0), 1300, 720, edgecolor='r', facecolor='None')
    ax.add_patch(rect)
    for j in range(len(pred[i])):
        plt.scatter(pred_gc_org[i][j][0], pred_gc_org[i][j][1], marker='o', color=colors[3], label=f'Pred')
        plt.scatter(pred_gc_zoomed[i][j][0], pred_gc_zoomed[i][j][1], marker='o', color=colors[2], label=f'Zoomed Pred')
        plt.scatter(refine_pred_copy[i][j][0], refine_pred_copy[i][j][1], marker='o', color=colors[0], label=f'SC Pred')
        plt.scatter(ground_truth[i][j][0], ground_truth[i][j][1], marker='x',color = colors[1], label=f'True')
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc_org[i][j][0] - ground_truth[i][j][0],
                  pred_gc_org[i][j][1] - ground_truth[i][j][1], color=colors[3], alpha=0.5)
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], pred_gc_zoomed[i][j][0] - ground_truth[i][j][0],
                  pred_gc_zoomed[i][j][1] - ground_truth[i][j][1], color=colors[2], alpha=0.5)
        plt.arrow(ground_truth[i][j][0], ground_truth[i][j][1], refine_pred_copy[i][j][0] - ground_truth[i][j][0],
                  refine_pred_copy[i][j][1] - ground_truth[i][j][1], color=colors[0], alpha=0.5)

        if (j == 0):
            plt.legend()
    plt.title('photos_result_' + condition_label[i] + '_x_vs_y', fontsize=15)
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    dst_filename = r'photo_test_plot/photos_result_' + condition_label[i] + r'_x_vs_y' + r'.jpg'
    plt.savefig(dst_filename)
    plt.clf()
