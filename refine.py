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


def rescale(pred_orginal,size):
    for i in range(len(pred_orginal)):
        for j in range(len(pred_orginal[i])):
            if i % 2 == 0 or i >= 46:
                pred_orginal[i][j][0] = pred_orginal[i][j][0] * size[0] / 1600
                pred_orginal[i][j][1] = pred_orginal[i][j][1] * size[1] / 825
            else:
                pred_orginal[i][j][0] = pred_orginal[i][j][0] * size[0] / 1300
                pred_orginal[i][j][1] = pred_orginal[i][j][1] * size[1] / 720
    return pred_orginal

def rev_rescale(pred_scaled,pred_size):
    for i in range(len(pred_scaled)):
        for j in range(len(pred_scaled[i])):
            if i % 2 == 0 or i >= 46:
                pred_scaled[i][j][0] = pred_scaled[i][j][0] * 1600 / pred_size[0]
                pred_scaled[i][j][1] = pred_scaled[i][j][1] * 825 / pred_size[1]
            else:
                pred_scaled[i][j][0] = pred_scaled[i][j][0] * 1300 / pred_size[0]
                pred_scaled[i][j][1] = pred_scaled[i][j][1] * 720 / pred_size[1]
    return pred_scaled

def margin(pred,size,margin_size=0.2):
    for i in range(len(pred)):
        for j in range(len(pred[i])):
                pred[i][j][0] = pred[i][j][0] + size[0] * margin_size
                pred[i][j][1] = pred[i][j][1] + size[1] * margin_size
    size_with_margin=np.multiply(size,(1+2*margin_size))
    return pred,size_with_margin

def remove_margin(pred,size_with_margin,margin_size=0.2):
    for i in range(len(pred)):
        for j in range(len(pred[i])):
                pred[i][j][0] = pred[i][j][0] - size_with_margin[0] * margin_size / (1 + margin_size)
                pred[i][j][1] = pred[i][j][1] - size_with_margin[1] * margin_size / (1 + margin_size)
    size=np.multiply(size_with_margin,(1/(1+2*margin_size)))
    return pred,size


def Revert_normalization(pred_pitchyaw,RMat):
    pred_vector = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], []]
    for i in range(len(pred_pitchyaw)):
        pred_vector[i] = warp_norm.pitchyaw_to_vector(pred_pitchyaw[i])

    org_pred = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(len(pred_vector)):
        for j in range(len(pred_vector[i])):
            # print(RMat[i][j])
            org_pred[i].append(np.dot(np.linalg.inv(RMat[i][j]), pred_vector[i][j].T))
    # print(org_pred[0][0])

    pixel_scale_tan = np.array([0.202, 0.224])
    pixel_scale_chen = np.array([0.22, 0.235])

    pred_gc = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
               [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(len(org_pred)):
        for j in range(len(org_pred[i])):
            if i % 2 == 0 or i >= 46:
                pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale_tan, face_center=-600))
            else:
                pred_gc[i].append(warp_norm.vector_to_gc(org_pred[i][j], pixel_scale_chen, face_center=-600))

    org_tan = np.array([800, 0])  # tan 1600*825
    org_chen = np.array([650, 0])  # chen 1300*720

    pred_gc_org = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], []]
    for i in range(52):
        if i % 2 == 0 or i >= 46:
            pred_gc_org[i] = org_tan + pred_gc[i]
        else:
            pred_gc_org[i] = org_chen + pred_gc[i]

    return pred_gc_org

def PoG_errors(pred, ground_truth):
    pixel_scale_tan = np.array([0.202, 0.224])
    pixel_scale_chen = np.array([0.22, 0.235])
    pred_xerrors=[]
    pred_yerrors=[]
    for i in range(52):
        total_xerrors = 0
        total_yerrors = 0
        for j in range(len(pred[i])):
            total_xerrors=total_xerrors+abs(pred[i][j][0]-ground_truth[i][j][0])
            total_yerrors = total_yerrors + abs(pred[i][j][1] - ground_truth[i][j][1])
        pred_xerrors.append(total_xerrors / (len(pred[i])))
        pred_yerrors.append(total_yerrors / (len(pred[i])))
    pred_xerrors_cm=[]
    pred_yerrors_cm=[]
    for i in range(52):
        if i % 2 == 0 or i >= 46:
            pred_xerrors_cm.append(pred_xerrors[i] * 0.1 * pixel_scale_tan[0])
            pred_yerrors_cm.append(pred_yerrors[i] * 0.1 * pixel_scale_tan[1])
        else:
            pred_xerrors_cm.append(pred_xerrors[i] * 0.1 * pixel_scale_chen[0])
            pred_yerrors_cm.append(pred_yerrors[i] * 0.1 * pixel_scale_chen[1])
    return pred_xerrors_cm, pred_yerrors_cm

def in_screen_percentage(pred):
    in_screen_net = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                     [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                     [], []]
    for i in range(52):
        for j in range(len(pred[i])):
            if i % 2 == 0 or i >= 46:  # tan
                if (0 - 0.05 * 1600 <= pred[i][j][0] <= 1600 + 0.05 * 1600 and 0 - 0.05 * 825 <=
                        pred[i][j][1] <= 825 + 0.05 * 825):
                    in_screen_net[i].append(1)
                else:
                    in_screen_net[i].append(0)
            else:
                if (0 - 0.05 * 1300 <= pred[i][j][0] <= 1300 + 0.05 * 1300 and 0 - 0.05 * 720 <=
                        pred[i][j][1] <= 720 + 0.05 * 720):
                    in_screen_net[i].append(1)
                else:
                    in_screen_net[i].append(0)
    # print(in_screen)

    in_screen_percentage = []
    for i in range(52):
        total_in = 0
        for j in range(len(pred[i])):
            total_in = total_in + in_screen_net[i][j]
        in_screen_percentage.append(total_in / len(pred[i]))
    return in_screen_percentage

def Zoom(pred_history, ground_truth_history, pred, range_scale = 0.8):
    zoom_scale = []
    for i in range(len(pred_history)):
        pred_percentile = np.percentile(pred_history[i], [100*(1-range_scale)/2, 100-100*(1-range_scale)/2], axis=0)  # get percentile
        truth_percentile = np.percentile(ground_truth_history[i], [100*(1-range_scale)/2, 100-100*(1-range_scale)/2], axis=0)  # get percentile
        pred_percentile_range = pred_percentile[1] - pred_percentile[0]
        truth_percentile_range = truth_percentile[1] - truth_percentile[0]
        zoom_scale.append(truth_percentile_range / pred_percentile_range)
    # print(zoom_scale)
    pred_gc_zoomed = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                      [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                      [], [], [], []]
    for i in range(len(pred)):
        total_pred = [0, 0]
        for j in range(int(len(pred[i]))):
            total_pred = total_pred + pred[i][j]
        center = total_pred / int(len(pred[i]))
        pred_gc_zoomed[i] = (pred[i] - center) * zoom_scale[i] + center
    return pred_gc_zoomed

def Self_Calibration(pred_history, ground_truth_history, pred):
    gtr = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
           [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    aver_pred = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                 [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    offset = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
              [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(len(pred_history)):
        total_truth = [0, 0]
        total_pred = [0, 0]
        if i % 2 == 0 or i >= 46:
            gtr[i] = [800, 412.5]
        else:
            gtr[i] = [650, 360]

        for j in range(int(len(pred_history[i]))):
            total_truth = total_truth + ground_truth_history[i][j]
            total_pred = total_pred + pred_history[i][j]
        aver_pred[i] = total_pred / int(len(pred_history[i]))
        offset[i] = aver_pred[i] - gtr[i]
    # print(gtr)
    # print(aver_pred)
    # print(offset)

    refine_pred = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], []]
    for i in range(len(pred)):
        refine_pred[i] = pred[i] - offset[i]
    return refine_pred

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