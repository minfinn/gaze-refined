import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
from tqdm import tqdm
import data_loader
import trainer
import argparse
from config import get_config

if __name__ == '__main__':
    config, unparsed = get_config()
    config.is_train = False
    config.batch_size =10
    config.use_gpu = True
    config.ckpt_dir = './ckpt'
    config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    test_data = data_loader.get_test_loader('D:/Deeplearning/dataset/Dataset_com/preprocessed_labels.csv',
                                            batch_size=config.batch_size,
                                            data_path= 'D:/Deeplearning/dataset/Dataset_com/preprocessed_images',
                                            num_workers=0)
    xgaze = trainer.Trainer(config, test_data)
    xgaze.test(save_fname_pkl='gaze_pred_old.pkl',save_fname_txt='results_old.txt')