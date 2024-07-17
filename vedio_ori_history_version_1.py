

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



if __name__ == '__main__':
    import cv2
    import os
    import pandas as pd
    import pickle
    # from refine_pred import
    # from gaze_classification import
    import warp_norm
    import numpy as np
    import data_loader
    import trainer
    from config import get_config
    from pt_module import StNet, StRefine
    from ipdb import set_trace as st
    import numpy as np
    import data_loader
    import trainer
    import argparse
    from config import get_config

    # def calculate_average_distance(coords):
    #     distances = [np.linalg.norm(np.array(coord) - np.array(coords[i - 1])) for i, coord in
    #                  enumerate(coords[1:], start=1)]
    #     return np.mean(distances)
    #
    #
    # def mark_invalid_coordinates(coords, labels, threshold=1):
    #     valid_segments = []
    #     current_segment = []
    #
    #     for coord, label in zip(coords, labels):
    #         if label == 1:
    #             current_segment.append(coord)
    #         elif current_segment:
    #             valid_segments.append(current_segment)
    #             current_segment = []
    #
    #     if current_segment:
    #         valid_segments.append(current_segment)
    #
    #     # 计算每个有效段的平均坐标和平均距离
    #     average_coords = [np.mean(segment, axis=0) for segment in valid_segments]
    #     average_distance = calculate_average_distance(coords)
    #
    #     # 将与平均距离大于阈值的坐标标记为无效（0）
    #     for i, segment in enumerate(valid_segments):
    #         for coord in segment:
    #             distance = np.linalg.norm(coord - average_coords[i])
    #             if distance > threshold:
    #                 labels[coords.index(coord)] = 0
    #
    #     return labels
    #
    #
    # # 示例用法
    # coordinates = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
    # labels = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1]
    #
    # result_labels = mark_invalid_coordinates(coordinates, labels, threshold=80)
    # print(result_labels)




    cam_tan = 'D:/Deeplearning/dataset/Calibration/Calibration/camTan.xml'  # this is camera calibration information file obtained with OpenCV
    fs_tan = cv2.FileStorage(cam_tan, cv2.FILE_STORAGE_READ)
    w_tan = 1600
    h_tan = 825
    pixel_scale_tan = 0.211667
    camera_matrix_tan = fs_tan.getNode(
        'Camera_Matrix').mat()  # camera calibration information is used for data normalization
    camera_distortion_tan = fs_tan.getNode('Distortion_Coefficients').mat()

    cam_chen = 'D:/Deeplearning/dataset/Calibration/Calibration/camChen.xml'  # this is camera calibration information file obtained with OpenCV
    fs_chen = cv2.FileStorage(cam_chen, cv2.FILE_STORAGE_READ)
    w_chen = 1300
    h_chen = 720
    pixel_scale_chen = 0.223427
    camera_matrix_chen = fs_chen.getNode(
        'Camera_Matrix').mat()  # camera calibration information is used for data normalization
    camera_distortion_chen = fs_chen.getNode('Distortion_Coefficients').mat()



    model1, model2, model3 = warp_norm.xmodel()#face detector



    model_dir = 'D:/Deeplearning/GazeNormalization-main/models'
    state_name = 'spatical_transform_model_fake_eyetracking_dataset_1kg_128_72_error00_valid_random_ii_in_sequence_person_gt_lr_0.1_99_full.pt'
    state_path = model_dir + '/' + state_name

    save_preprocessed_dir = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/video_preprocessed_images_smooth'
    os.makedirs(save_preprocessed_dir, exist_ok=True)

    save_image_dir = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/video_images'
    os.makedirs(save_image_dir, exist_ok=True)


    iris_save_path = "./iris_test"
    os.makedirs(iris_save_path, exist_ok=True)
    img_path = ['D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/1.avi']

    dataset = []
    imgs = []
    filenames = []
    face_area_label = []
    num = 0
    fileend = '.jpg'
    for i in range(len(img_path)):
        cap = cv2.VideoCapture(img_path[i])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(width,height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #
        # outVideo = cv2.VideoWriter(os.path.join(iris_save_path, str(i) + ".mp4"), fourcc, fps, (width, height))


        while True:
            success,img=cap.read()
            if not success:
                break
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # st()
            # img=img[:,:img.shape[1]//2,:]
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            # st()
            print(img.shape)
            num = num + 1
            filename = str(num) + fileend
            save_path = os.path.join(save_image_dir,f'video_image_{filename}')
            cv2.imwrite(save_path, img)
            filenames.append(filename)
            imgs.append(img)
    # imgs=np.array(imgs)
    # print(imgs.shape)
    labels = np.full((len(imgs), 2), -1)
    images, gaze_center, R, Ear, face_center_in_img = warp_norm.GazeNormalization_video_version2(imgs, camera_matrix_chen, camera_distortion_chen, labels, w_chen, h_chen,
                                                             predictor=model1, face_detector=model2, eve_detector=model3, filter_over=5)

    # print(images.shape)
    print(R[0])
    for i in range(len(images)):
        scale = np.array([[1, 1, 1], [1, 1, 1], [0.8, 0.8, 0.8]])
        R[i] = R[i] * scale
        print(filenames[i])
        preprocessed_save_path = os.path.join(save_preprocessed_dir, f'video_preprocessed_image_smooth_{filenames[i]}')
        cv2.imwrite(preprocessed_save_path, images[i])
        face_area_label = get_face_center_label(face_center_in_img[i][0], face_center_in_img[i][1], (320, 240))
        dataset.append({'image_path': f'video_preprocessed_image_smooth_{filenames[i]}', 'original_label': labels[i], 'R': R[i], 'Ear':Ear[i], 'face_area_label':face_area_label} )

    print(R[0])
    pickle_file_path = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/dataset_dict_smooth.pkl'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(dataset, file)

    # 保存标签为CSV
    csv_file_path = 'D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/preprocessed_labels_smooth.csv'
    df = pd.DataFrame(dataset)
    df.to_csv(csv_file_path, index=False)

    print('Preprocessing and saving complete.')





    #predict gaze
    config, unparsed = get_config()
    config.is_train = False
    config.batch_size = 10
    config.use_gpu = True
    config.ckpt_dir = './ckpt'
    config.pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
    test_data = data_loader.get_test_loader('D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/preprocessed_labels_smooth.csv',data_path='D:/Deeplearning/dataset/Dataset_video2_chen/Chen_glass/Video/video_preprocessed_images_smooth',
                                            batch_size=config.batch_size, num_workers=0)
    xgaze = trainer.Trainer(config, test_data)
    xgaze.test(save_fname_pkl='gaze_pred_smooth.pkl',save_fname_txt='results_smooth.txt')
    #Todo:1.整套流程应用在原来的dataset，history采用本condition前50%的data，看效果    2.小模型版本

    # 跨设备的图
    #Todo:1.SC模块前根据history做range校正模块 2.function 3.预处理过程的问题 4.图片plot缺少跨设备的condition
