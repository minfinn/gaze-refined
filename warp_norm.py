import sys

sys.path.append("./FaceAlignment")
import face_alignment
from imutils import face_utils
import cv2
import dlib
import numpy as np
from ipdb import set_trace as st

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def estimateHeadPose(landmarks, face_model, camera,
                     distortion=np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]),
                     iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec


def xmodel():
    '''预读取模型'''
    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    return predictor, face_detector, fa


# 工具
def calculate_face_area(face):
    min_x, max_x = np.min(face[0]), np.max(face[0])
    min_y, max_y = np.min(face[1]), np.max(face[1])
    area = (max_x - min_x) * (max_y - min_y)
    return area

def xnorm(input, camera_matrix,
          camera_distortion=np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]),
          predictor=dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat'),
          face_detector=dlib.get_frontal_face_detector()):
    # face detection
    detected_faces = face_detector(cv2.cvtColor(input, cv2.COLOR_BGR2RGB), 1)  ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face by dlib')
        hr = np.zeros((1, 3))
        ht = np.zeros((1, 3))
        return hr, ht, -1, (-1,-1)
    print('detected one face')

    largest_face = max(detected_faces, key=lambda rect: rect.width() * rect.height())
    print("max face position:", largest_face)

    # shape = predictor(input, detected_faces[idx])
    shape = predictor(input, largest_face)

    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)

    Ear = []
    for i in range(2):
        Ear.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
            landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / (
                               2 * np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i], 2)))
    Ear = np.mean(np.asarray(Ear))

    # load face model
    face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)

    # 检测得到的人脸标定点，将其调整为对应3D点的形式
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    face_center_in_img = np.mean(landmarks_sub, axis=0)
    print(face_center_in_img.shape)
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
    return hr, ht, Ear, face_center_in_img

def xnorm_smooth(input, camera_matrix, input_idx, landmarks_sub_history, filter_over=5,
          camera_distortion=np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]),
          predictor=dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat'),
          face_detector=dlib.get_frontal_face_detector()):
    # face detection
    print('input_idx:',input_idx)
    detected_faces = face_detector(cv2.cvtColor(input, cv2.COLOR_BGR2RGB), 1)  ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face by dlib')
        hr = np.zeros((3, 1))
        ht = np.zeros((3, 1))
        return hr, ht, -1,landmarks_sub_history,(-1,-1)
    print('detected one face')

    largest_face = max(detected_faces, key=lambda rect: rect.width() * rect.height())
    print("max face position:", largest_face)

    # shape = predictor(input, detected_faces[idx])
    shape = predictor(input, largest_face)

    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)

    Ear = []
    for i in range(2):
        Ear.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2) + np.linalg.norm(
            landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)) / (
                               2 * np.linalg.norm(landmarks[36 + 6 * i] - landmarks[39 + 6 * i], 2)))
    Ear = np.mean(np.asarray(Ear))

    # load face model
    face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(6, 1, 3)

    # 检测得到的人脸标定点，将其调整为对应3D点的形式
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    landmarks_sub_filtered = filter_landmarks(landmarks_sub, landmarks_sub_history, input_idx, filter_over)
    face_center_in_img=np.mean(landmarks_sub_filtered,axis=0)[0]
    print('face_center_in_img:',face_center_in_img)
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
    return hr, ht, Ear,landmarks_sub_history,face_center_in_img

def filter_landmarks(landmarks_sub, landmarks_sub_history, input_idx, filter_over):
    if input_idx >= filter_over-1:
        landmarks_sub_history=np.append(landmarks_sub_history[1:filter_over],[landmarks_sub],axis=0)
    else:
        landmarks_sub_history[input_idx+1]=landmarks_sub
        return landmarks_sub
    # landmarks_sub_filtered=np.full((6, 1, 2),0)
    landmarks_sub_filtered=np.mean(landmarks_sub_history,axis=0)
    return landmarks_sub_filtered
def xnorm_68(input, camera_matrix,
             camera_distortion=np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]),
             predictor=dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat'),
             face_detector=dlib.get_frontal_face_detector()):
    # face detection
    detected_faces = face_detector(cv2.cvtColor(input, cv2.COLOR_BGR2RGB), 1)  ## convert BGR image to RGB for dlib
    if len(detected_faces) == 0:
        print('warning: no detected face')
        hr = np.zeros((1, 3))
        ht = np.zeros((1, 3))
        return hr, ht
    print('detected one face')
    shape = predictor(input, detected_faces[
        0])  ## only use the first detected face (assume that each input image only contains one face)
    shape = face_utils.shape_to_np(shape)
    landmarks = []
    for (x, y) in shape:
        landmarks.append((x, y))
    landmarks = np.asarray(landmarks)
    arr1 = np.array([i for i in range(100)])
    result = np.concatenate([arr1[8:9], arr1[17:60], arr1[61:64], arr1[65:68]])
    landmarks_sub = landmarks[result, :]
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(50, 1, 2)  # input to solvePnP requires such shape
    # load face model
    face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    # landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    # landmarks = landmarks.reshape(-1, 2)
    # head_pose_estimator = HeadPoseEstimator()
    # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix[cam_id])
    ## the easy way to get head pose information, fast and simple
    facePts = face_model.reshape(50, 1, 3)

    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
    return hr, ht


def enorm(input, camera_matrix,
          camera_distortion=np.array([-0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927]),
          eve_detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)):
    '''
    input: 待处理的图片，可以不经过人脸提取
    camera_matrix: 相机内参矩阵
    camera_distortion: 相机畸变矩阵，这里的默认值是xGaze的相机参数
    '''
    # 标定人脸关键点
    preds = eve_detector.get_landmarks(input)
    if preds is None:
        print('warning: no detected face by fa')
        hr = np.zeros((3, 1))
        ht = np.zeros((3, 1))
        return hr, ht, -1, (-1,-1)
    # 左右眼角和嘴角
    landmark_use = [36, 39, 42, 45, 48, 54]
    max_face_index = np.argmax(np.apply_along_axis(calculate_face_area, axis=1, arr=preds))
    lm = preds[max_face_index]
    lm = lm[landmark_use, :]
    face = np.loadtxt('./modules/faceModelGeneric.txt')
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 3)
    # fid = cv2.FileStorage('./data-preprocessing-gaze//data/calibration/cameraCalib.xml', cv2.FileStorage_READ)
    # camera_matrix = fid.getNode("camera_matrix").mat()
    # camera_distortion = fid.getNode("cam_distortion").mat()
    lm = lm.astype(np.float32)
    lm = lm.reshape(num_pts, 1, 2)
    face_center_in_img = np.mean(lm,axis=0)
    print(face_center_in_img.shape)
    hr, ht = estimateHeadPose(lm, facePts, camera_matrix, camera_distortion)
    # Ear未完成，占位
    Ear = -1
    return hr, ht, Ear, face_center_in_img

    # normalization function for the face images


def xtrans(img, face_model, hr, ht, cam, w=1920, h=1080, gc=np.array([100, 100]), pixel_scale=np.array([0.215, 0.215])):
    '''
    img: 人脸图片
    face_model: 人脸模型,[68,2]
    hr: 来自annotation, 旋转向量, [3,1]
    ht: 来自annotation, 平移向量, [3,1]
    cam: Camera_Matrix
    gc: 来自annotation, gaze point on the screen coordinate system, [2,1]
    '''
    # normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between face and camera
    roiSize = (224, 224)  # size of cropped image
    # compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    if (gc.size == 2):
        # should change depend on the camera position
        x = -gc[0] + w / 2
        y = -gc[1] + h
        gc = np.array([x, y])
        gc = gc * pixel_scale
        gc = np.r_[gc, np.array([0])]
        gc = gc.reshape((3, 1))
    # 将二维人脸（原图）套入三维模型
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix, [3,3]
    Fc = np.dot(hR, face_model.T) + ht  # [3,50]
    # 取得人脸中心
    # two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    # mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(Fc, axis=1).reshape((3, 1))
    # print('face_center:',face_center)
    # normalize image
    distance = np.linalg.norm(face_center)  # actual distance between and original camera
    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    # forward *= 0.7

    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # image normalization

    # normalize rotation
    hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize gaze vector
    # 以下已废弃
    gc_normalized = gc - face_center  # gaze vector
    gc_normalized = np.dot(R, gc_normalized)  # 这里只追求旋转，所以没有与相机矩阵相乘
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)  # 归一化
    gc_normalized = -gc_normalized
    return img_warped, hr_norm, gc_normalized, R


def draw_norm_gaze(image_in, gc_normalized, thickness=2, color=(0, 0, 255)):
    '''Draw gaze angle on given image with a given eye positions.'''
    if (gc_normalized.size == 3):
        pitchyaw = vector_to_pitchyaw(np.array([gc_normalized]))[0]
    else:
        pitchyaw = gc_normalized
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:  # to draw on the image, we need to convert to RGB
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(int)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def draw_gaze(image, RMat, pred, PoG_refined, pixel_scale, device_result, is_refined, thickness=2, color=(0, 0, 255), data_normalized=[]):
    '''Draw gaze angle on given image with a given eye positions.'''
    # 1.Reduction
    org_tan = np.array([800, 0])  # tan 1600*825
    org_chen = np.array([650, 0])  # chen 1300*720
    gaze_vector_refined = gc_to_vector(PoG_refined - org_chen, pixel_scale)
    # print(gaze_vector)
    pred_refined = vector_to_pitchyaw(np.array([gaze_vector_refined]))
    pred_refined = pred_refined[0]

    gaze_vector = pitchyaw_to_vector(pred.reshape((1, 2)))
    # print(gaze_vector)
    PoG = vector_to_gc(gaze_vector[0], pixel_scale)
    PoG = org_chen + PoG


    if ((RMat == 0).all()):
        org_pred = gaze_vector
        org_pred_refined = gaze_vector_refined
    else:
        org_pred = np.dot(np.linalg.inv(RMat), gaze_vector.T)
        org_pred_refined = np.dot(np.linalg.inv(RMat), gaze_vector_refined.T)
    pitchyaw_original = vector_to_pitchyaw(org_pred.reshape((1, 3)))[0]
    pitchyaw_refined = vector_to_pitchyaw(org_pred_refined.reshape((1, 3)))[0]
    image_out = image
    # 2.Draw vector(original)
    r = 56
    pos = int(400 - 56)

    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = r * np.sin(theta_circle)
    y_circle = r * np.cos(theta_circle)
    theta_new = pitchyaw_original[1]
    phi_new = pitchyaw_original[0]

    # 根据phi值计算y和z坐标
    x = -r * np.cos(phi_new) * np.sin(theta_new)
    y = r * np.sin(phi_new)
    z = r * np.cos(phi_new) * np.cos(theta_new)
    # 根据真实坐标还原角度并投影
    # 绘制竖圆
    phi_1 = np.linspace(0, np.pi, 200)
    proj_v = np.array([x, z])
    x_axis = np.array([1, 0])
    cos_theta_1 = np.dot(proj_v, x_axis) / (np.linalg.norm(proj_v) * np.linalg.norm(x_axis))
    theta_1 = np.arccos(cos_theta_1)
    x_1 = r * np.sin(phi_1) * np.cos(theta_1)
    y_1 = r * np.cos(phi_1)

    # 绘制横圆
    phi_2 = np.linspace(0, np.pi, 200)
    proj_y = np.array([y, z])
    y_axis = np.array([1, 0])
    cos_theta_2 = np.dot(proj_y, y_axis) / (np.linalg.norm(proj_y) * np.linalg.norm(y_axis))
    theta_2 = np.arccos(cos_theta_2)
    x_2 = r * np.cos(phi_2)
    y_2 = r * np.sin(phi_2) * np.cos(theta_2)

    # 绘制点和线
    cv2.circle(image_out, (int(x + pos + 170), int(-y + pos)), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
    cv2.circle(image_out, (int(0 + pos + 170), int(0 + pos)), radius=1, color=(255, 255, 255), thickness=-1)  # 白色点
    cv2.line(image_out, (pos + 170, pos), (int(x + pos + 170), int(-y + pos)), color=(255, 255, 255), lineType=cv2.LINE_AA)  # 白色虚线

    # 绘制圆形投影（根据x_circle, y_circle的值绘制）
    for i in range(len(x_circle) - 1):
        cv2.line(image_out, (int(x_circle[i] + pos + 170), int(-y_circle[i] + pos)),
                 (int(x_circle[i + 1] + pos + 170), int(-y_circle[i + 1] + pos)), color=(0, 255, 0), thickness=1)
    for i in range(len(x_1) - 1):
        cv2.line(image_out, (int(x_1[i] + pos + 170), int(-y_1[i] + pos)), (int(x_1[i + 1] + pos + 170), int(-y_1[i + 1] + pos)),
                 (0, 0, 255), 2)
    for i in range(len(x_2) - 1):
        cv2.line(image_out, (int(x_2[i] + pos + 170), int(-y_2[i] + pos)), (int(x_2[i + 1] + pos + 170), int(-y_2[i + 1] + pos)),
                 (0, 0, 255), 2)

    # 2.Draw vector(refined)
    r = 56
    pos = int(400 - 56)

    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = r * np.sin(theta_circle)
    y_circle = r * np.cos(theta_circle)
    theta_new = pitchyaw_refined[1]
    phi_new = pitchyaw_refined[0]

    # 根据phi值计算y和z坐标
    x = -r * np.cos(phi_new) * np.sin(theta_new)
    y = r * np.sin(phi_new)
    z = r * np.cos(phi_new) * np.cos(theta_new)
    # 根据真实坐标还原角度并投影
    # 绘制竖圆
    phi_1 = np.linspace(0, np.pi, 200)
    proj_v = np.array([x, z])
    x_axis = np.array([1, 0])
    cos_theta_1 = np.dot(proj_v, x_axis) / (np.linalg.norm(proj_v) * np.linalg.norm(x_axis))
    theta_1 = np.arccos(cos_theta_1)
    x_1 = r * np.sin(phi_1) * np.cos(theta_1)
    y_1 = r * np.cos(phi_1)

    # 绘制横圆
    phi_2 = np.linspace(0, np.pi, 200)
    proj_y = np.array([y, z])
    y_axis = np.array([1, 0])
    cos_theta_2 = np.dot(proj_y, y_axis) / (np.linalg.norm(proj_y) * np.linalg.norm(y_axis))
    theta_2 = np.arccos(cos_theta_2)
    x_2 = r * np.cos(phi_2)
    y_2 = r * np.sin(phi_2) * np.cos(theta_2)

    # 绘制点和线
    cv2.circle(image_out, (int(x + pos + 170), int(-y + pos)), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点
    cv2.circle(image_out, (int(0 + pos + 170), int(0 + pos)), radius=1, color=(255, 255, 255), thickness=-1)  # 白色点
    cv2.line(image_out, (pos + 170, pos), (int(x + pos + 170), int(-y + pos)), color=(255, 255, 255),
             lineType=cv2.LINE_AA)  # 黑色虚线

    # 绘制圆形投影（根据x_circle, y_circle的值绘制）
    for i in range(len(x_circle) - 1):
        cv2.line(image_out, (int(x_circle[i] + pos + 170), int(-y_circle[i] + pos)),
                 (int(x_circle[i + 1] + pos + 170), int(-y_circle[i + 1] + pos)), color=(0, 255, 0), thickness=1)
    for i in range(len(x_1) - 1):
        cv2.line(image_out, (int(x_1[i] + pos + 170), int(-y_1[i] + pos)),
                 (int(x_1[i + 1] + pos + 170), int(-y_1[i + 1] + pos)),
                 (0, 255, 0), 2)
    for i in range(len(x_2) - 1):
        cv2.line(image_out, (int(x_2[i] + pos + 170), int(-y_2[i] + pos)),
                 (int(x_2[i + 1] + pos + 170), int(-y_2[i + 1] + pos)),
                 (0, 255, 0), 2)
    if device_result==1:
        text = 'looking at the screen'
    else:
        text = 'No looking at the screen'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    font_color = (255, 255, 255)  # 白色
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (image_out.shape[1] - text_size[0] - 100, image_out.shape[0] - 20)
    cv2.putText(image_out, text, text_position, font, font_scale, font_color, font_thickness)
    text2 = 'Refined Gaze Center:{:2.2f}'.format(PoG_refined[0])+',{:2.2f}'.format(PoG_refined[1])
    text_position2 = (image_out.shape[1] - text_size[0] - 100, image_out.shape[0] - 30)
    cv2.putText(image_out, text2, text_position2, font, font_scale, font_color, font_thickness)
    text3 = 'Refined Gaze Pitchyaw:{:2.2f}'.format(pitchyaw_refined[0]) + ',{:2.2f}'.format(pitchyaw_refined[1])
    text_position3 = (image_out.shape[1] - text_size[0] - 100, image_out.shape[0] - 40)
    cv2.putText(image_out, text3, text_position3, font, font_scale, font_color, font_thickness)
    text4 = 'Original Gaze Center:{:2.2f}'.format(PoG[0]) + ',{:2.2f}'.format(PoG[1])
    text_position4 = (image_out.shape[1] - text_size[0] - 100, image_out.shape[0] - 50)
    cv2.putText(image_out, text4, text_position4, font, font_scale, font_color, font_thickness)
    text5 = 'Original Gaze Pitchyaw:{:2.2f}'.format(pitchyaw_original[0]) + ',{:2.2f}'.format(pitchyaw_original[1])
    text_position5 = (image_out.shape[1] - text_size[0] - 100, image_out.shape[0] - 60)
    cv2.putText(image_out, text5, text_position5, font, font_scale, font_color, font_thickness)
    return image_out

def vector_to_gc(gv, pixel_scale, face_center=-600):
    '''实现向量和屏幕注视点的转换，转换的结果为相对于摄像头原点的坐标'''
    # gv为直角坐标系注视向量
    # 坐标系方向转换至与屏幕坐标系一致(反转y轴)
    gv[1] = -gv[1]
    z = np.array([0, 0, face_center])
    theta = np.arcsin(np.linalg.norm(np.cross(gv, z)) / (np.linalg.norm(gv) * np.linalg.norm(z)))
    scale = np.linalg.norm(z) / (np.cos(theta) * np.linalg.norm(gv))
    gp = scale * gv + z  # 单位为mm
    print("vector_to_gc: vector = ", gp)
    gp = np.delete(gp, 2, axis=0)
    gp = gp / pixel_scale
    return gp

def gc_to_vector(gp, pixel_scale, face_center=-600):
    '''将屏幕注视点转换为注视向量'''
    # 将二维注视坐标转换为毫米单位
    gp = gp * pixel_scale
    # 构建三维注视点，z坐标为face_center
    gp_3d = np.array([gp[0], gp[1], 0])
    # 计算注视向量（从原点到注视点）
    gv = gp_3d + np.array([0, 0, face_center])
    # 调整y轴方向
    gv[1] = -gv[1]
    return gv

def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi


def GazeNormalization(image, camera_matrix, camera_distortion, gc, w, h, predictor, face_detector='None',
                      eve_detector='None', method='xgaze'):
    if (method == 'xgaze'):
        hr, ht, Ear, face_center = xnorm(image, camera_matrix, camera_distortion, predictor, face_detector)
        if (hr.all() == 0 and ht.all() == 0):
            hr, ht, Ear, face_center = enorm(image, camera_matrix, camera_distortion, eve_detector)
            if (hr.all() == 0 and ht.all() == 0):
                warp_image = np.zeros((224, 224, 3), dtype=np.byte)
                gcn = np.zeros((3, 1))
                R = np.zeros((3, 3))
                return warp_image, gcn, R, -1, (-1,-1)
            face = np.loadtxt('./modules/faceModelGeneric.txt')
            num_pts = face.shape[1]
            face_model = face.T.reshape(num_pts, 3)
            warp_image, _, gcn, R = xtrans(image, face_model, hr, ht, camera_matrix, w, h, gc)
            return warp_image, gcn, R, Ear, face_center[0]
        face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
        landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
        face_model = face_model_load[landmark_use, :]
        warp_image, _, gcn, R = xtrans(image, face_model, hr, ht, camera_matrix, w, h, gc)
    return warp_image, gcn, R, Ear, face_center[0]

def avgSmooth(arr, avgOver=5, preserveLength=False):
    ''' example:
        datAvg = np.apply_along_axis(lambda x: avgFilter(x, avgOver=5), 0, dat)'''
    arrAvg = []
    i = 0
    while i + avgOver <= len(arr):
        oneArr = arr[i: i + avgOver]
        avg = np.mean(oneArr)
        arrAvg.append(avg)
        i += 1
    if preserveLength:
        arrAvg = [arr[0], ] * (avgOver - 1) + arrAvg
    return arrAvg

def GazeNormalization_video_version1(image, camera_matrix, camera_distortion, gc, w, h, predictor, face_detector='None',
                      eve_detector='None', method='xgaze', filter_over=5):
    '''ht,hr are filtered in GazeNormalization_video_version1'''
    if (method == 'xgaze'):
        hr = np.zeros((len(image), 3, 1))
        ht = np.zeros((len(image), 3, 1))
        warp_image = []
        for i in range(len(image)):
            warp_image.append(np.zeros((224, 224, 3), dtype=np.byte))
        gcn = np.zeros((len(image), 3, 1))
        R = np.zeros((len(image), 3, 3))
        Ear = np.zeros((len(image)))
        for i in range(len(image)):
            hr[i], ht[i], Ear[i] = xnorm(image[i], camera_matrix, camera_distortion, predictor, face_detector)
            # filter/avgsmooth
            filter_cover=np.zeros((2,3,filter_over))
            if i < filter_over - 1:
                for j in range(3):
                    filter_cover[0][j][i+1]=hr[i][j]
                    filter_cover[1][j][i+1]=ht[i][j]
            else:
                for j in range(3):
                    filter_cover[0][j] = np.hstack((filter_cover[0][j][1:5], hr[i][j]))
                    filter_cover[1][j] = np.hstack((filter_cover[1][j][1:5], ht[i][j]))
                    hr[i][j] = np.mean(filter_cover[0][j])
                    ht[i][j] = np.mean(filter_cover[1][j])


            if (hr[i].all() == 0 and ht[i].all() == 0):
                hr[i], ht[i], Ear[i] = enorm(image[i], camera_matrix, camera_distortion, eve_detector)
                if (hr[i].all() == 0 and ht[i].all() == 0):
                    warp_image[i] = np.zeros((224, 224, 3), dtype=np.byte)
                    gcn[i] = np.zeros((3, 1))
                    R[i] = np.zeros((3, 3))
                    Ear[i]=-1
                    # return warp_image, gcn, R, -1
                else:
                    face = np.loadtxt('./modules/faceModelGeneric.txt')
                    num_pts = face.shape[1]
                    face_model = face.T.reshape(num_pts, 3)
                    warp_image[i], _, gcn[i], R[i] = xtrans(image[i], face_model, hr[i], ht[i], camera_matrix, w, h, gc[i])
                    # return warp_image, gcn, R, Ear
            else:
                face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
                landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
                face_model = face_model_load[landmark_use, :]
                warp_image[i], _, gcn[i], R[i] = xtrans(image[i], face_model, hr[i], ht[i], camera_matrix, w, h, gc[i])
                # cv2.imshow('img', warp_image[i])
                # st()
    # elif (method == 'xgaze68'):
    #     hr, ht = xnorm_68(image, camera_matrix, camera_distortion, predictor, face_detector)
    #     face_model = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
    #     warp_image, _, gcn, _ = xtrans(image, face_model, hr, ht, camera_matrix, gc)
    # else:
    #     hr, ht, Ear = enorm(image, camera_matrix, camera_distortion)
    #     face = np.loadtxt('./modules/faceModelGeneric.txt')
    #     num_pts = face.shape[1]
    #     face_model = face.T.reshape(num_pts, 3)
    #     warp_image, _, gcn, R = xtrans(image, face_model, hr, ht, camera_matrix, w, h, gc)
    return warp_image, gcn, R, Ear

def GazeNormalization_video_version2(image, camera_matrix, camera_distortion, gc, w, h, predictor, face_detector='None',
                      eve_detector='None', method='xgaze', filter_over=5):
    '''landmarks_sub are filtered in GazeNormalization_video_version2'''
    if (method == 'xgaze'):
        hr = np.zeros((len(image), 3, 1))
        ht = np.zeros((len(image), 3, 1))
        warp_image = []
        for i in range(len(image)):
            warp_image.append(np.zeros((224, 224, 3), dtype=np.byte))
        gcn = np.zeros((len(image), 3, 1))
        R = np.zeros((len(image), 3, 3))
        Ear = np.zeros((len(image)))
        landmarks_sub_history = np.full((filter_over,6,1,2),0)
        face_center = np.zeros((len(image),2))
        for i in range(len(image)):
            hr[i], ht[i], Ear[i],landmarks_sub_history,face_center[i] = xnorm_smooth(image[i], camera_matrix, i, landmarks_sub_history, filter_over, camera_distortion, predictor, face_detector)
            # # filter/avgsmooth
            # filter_cover=np.zeros((2,3,filter_over))
            # if i < filter_over - 1:
            #     for j in range(3):
            #         filter_cover[0][j][i+1]=hr[i][j]
            #         filter_cover[1][j][i+1]=ht[i][j]
            # else:
            #     for j in range(3):
            #         filter_cover[0][j] = np.hstack((filter_cover[0][j][1:5], hr[i][j]))
            #         filter_cover[1][j] = np.hstack((filter_cover[1][j][1:5], ht[i][j]))
            #         hr[i][j] = np.mean(filter_cover[0][j])
            #         ht[i][j] = np.mean(filter_cover[1][j])

            # print(hr[i],ht[i])
            if (hr[i].all() == 0 and ht[i].all() == 0):
                hr[i], ht[i], Ear[i], face_center[i]= enorm(image[i], camera_matrix, camera_distortion, eve_detector)
                if (hr[i].all() == 0 and ht[i].all() == 0):
                    warp_image[i] = np.zeros((224, 224, 3), dtype=np.byte)
                    gcn[i] = np.zeros((3, 1))
                    R[i] = np.zeros((3, 3))
                    Ear[i]=-1
                    # return warp_image, gcn, R, -1
                else:
                    face = np.loadtxt('./modules/faceModelGeneric.txt')
                    num_pts = face.shape[1]
                    face_model = face.T.reshape(num_pts, 3)
                    warp_image[i], _, gcn[i], R[i] = xtrans(image[i], face_model, hr[i], ht[i], camera_matrix, w, h, gc[i])
                    # return warp_image, gcn, R, Ear
            else:
                face_model_load = np.loadtxt('./modules/face_model.txt')  # Generic face model with 3D facial landmarks
                landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
                face_model = face_model_load[landmark_use, :]
                warp_image[i], _, gcn[i], R[i] = xtrans(image[i], face_model, hr[i], ht[i], camera_matrix, w, h, gc[i])
    return warp_image, gcn, R, Ear, face_center