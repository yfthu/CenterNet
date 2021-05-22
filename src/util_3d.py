import os
import json
import cv2
import yaml
import pyquaternion
import math
import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from twodtobev import undistort_contours, IPM_contours, cam_intrinsic, cam_extrinsic, compute_box_bev

OBJECT_THRESHOLD = 0.3
IOU_THRESHOLD=0.3

def trucate_angle(alpha):
    # return -pi to pi
    while alpha > math.pi or alpha < -math.pi:
        if alpha > math.pi:
            alpha -= 2*math.pi
        if alpha < -math.pi:
            alpha += 2 * math.pi
    return alpha


class Heduo_2nd_batch_Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, pre_process_func, anno_dir):
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.anno_dir = anno_dir
        self.img_dir = opt.img_dir
        self.all_annos_file = os.listdir(self.anno_dir)

    def __getitem__(self, index):
        anno_json = json.load(open(os.path.join(self.anno_dir,self.all_annos_file[index]), 'r'))[0]
        if u'\u6709\u70b9\u4e91\u6846' not in anno_json.keys():
            print("无点云框")
            gt_ydyk = []
        else:
            gt_ydyk = anno_json[u'\u6709\u70b9\u4e91\u6846']
        gt_tensor = torch.zeros(size=(len(gt_ydyk), 5),requires_grad=False)
        for i in range(len(gt_ydyk)):
            gt_tensor[i][0] = gt_ydyk[i]['center']['x']
            gt_tensor[i][1] = gt_ydyk[i]['center']['y']
            gt_tensor[i][2] = gt_ydyk[i]['width']
            gt_tensor[i][3] = gt_ydyk[i]['height']
            gt_tensor[i][4] = trucate_angle(gt_ydyk[i]['rotation']['z'] - np.pi / 2)
            # x,y,l,w,rotation
        img_relative_path = '/'.join(anno_json['img'].split('/')[-3:])
        img_path = os.path.join(self.img_dir, img_relative_path)
        image = cv2.imread(img_path)
        images, meta = {}, {}

        assert self.opt.test_scales == [1.0]
        for scale in self.opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)
        return index, {'images': images, 'image': image, 'meta': meta,
                       'img_name':img_relative_path.split('/')[-1],'gt_tensor':gt_tensor}


    def __len__(self):
        return len(self.all_annos_file)

def load_camera_parameter():
    f = open("M01_20200527/Camera/In/CAMERA_FRONT_CENTER.yaml")
    intrinsic_yaml = yaml.load(f, Loader=yaml.FullLoader)

    f = open("M01_20200527/Camera/Ex/lidar_front_center.yaml")
    extrinsic_yaml1 = yaml.load(f, Loader=yaml.FullLoader)

    f = open("M01_20200527/Lidar/VLP16/output_extrinsic.yaml")
    extrinsic_yaml2 = yaml.load(f, Loader=yaml.FullLoader)

    f = open("M01_20200527/Novatel/novatel_extrinsic.yaml")
    extrinsic_yaml3 = yaml.load(f, Loader=yaml.FullLoader)

    f = open("M01_20200527/Camera/Ex/fc_cam_pandar.yaml")
    extrinsic_yaml4 = yaml.load(f, Loader=yaml.FullLoader)

    K, D, new_K = cam_intrinsic(intrinsic_yaml)
    _, _, ex1 = cam_extrinsic(extrinsic_yaml1)
    _, _, ex2 = cam_extrinsic(extrinsic_yaml2)
    _, _, ex3 = cam_extrinsic(extrinsic_yaml3)
    _, _, ex4 = cam_extrinsic(extrinsic_yaml4)

    bTc = ex3 * ex2 * ex1.I  # todo ziji
    return K, D, new_K, bTc, ex4



def bev_bbox_error(x, keypoints):
    rectangle = compute_box_bev(x)
    rec_dif = rectangle - keypoints
    rec_dif_2 = rec_dif ** 2
    error = np.sum(rec_dif_2)
    return error

def cal_mask(object1, width=1000, height=1000):
    pts1 = compute_box_bev(object1)
    pts1[:, 0] = (25 - pts1[:, 0]) * (width / 50)
    pts1[:, 1] = (50 + pts1[:, 1]) * (height / 50)
    im1 = np.zeros([height, width], dtype="uint8")
    mask1 = cv2.fillPoly(im1, np.array([pts1], dtype=np.int32), 255)
    return mask1

def cal_iou(object1, object2):
    mask1 = cal_mask(object1)
    mask2 = cal_mask(object2)
    mask_and = cv2.bitwise_and(mask1, mask2)
    mask_or = cv2.bitwise_or(mask1, mask2)

    or_area = np.sum(np.float32(np.greater(mask_or, 0)))
    and_area = np.sum(np.float32(np.greater(mask_and, 0)))
    iou = and_area / or_area

    return iou


def cal_pr_one_img(np_centernet_bev,np_refine_pred,np_gt):
    # 输入都是nx5
    tp, fp, tp2, fp2, fn1, fn2,gtp1,gtp2 = 0, 0, 0, 0,0,0,0,0
    if np_centernet_bev.shape[0] == 0 or np_gt.shape[0] == 0:
        return tp, fp, tp2, fp2, fn1, fn2,gtp1,gtp2

    for centernet_bev_object in np_centernet_bev:
        diff_center = (centernet_bev_object - np_gt)[:,:2] # nx2
        distance_center = [np.sqrt(x * x + y * y) for (x, y) in diff_center] # n
        min_index = distance_center.index(min(distance_center))
        iou = cal_iou(centernet_bev_object, np_gt[min_index])
        is_true = (iou >= IOU_THRESHOLD)
        tp += (is_true == True)
        fp += (is_true == False)



    for pred_object in np_refine_pred:
        diff_center = (pred_object - np_gt)[:, :2]  # nx2
        distance_center = [np.sqrt(x * x + y * y) for (x, y) in diff_center]  # n
        min_index = distance_center.index(min(distance_center))
        iou = cal_iou(pred_object, np_gt[min_index])
        is_true = (iou >= IOU_THRESHOLD)
        tp2 += (is_true == True)
        fp2 += (is_true == False)


    for gt_object in np_gt:
        diff_center = (gt_object - np_centernet_bev)[:, :2]  # nx2
        distance_center = [np.sqrt(x * x + y * y) for (x, y) in diff_center]  # n
        min_index = distance_center.index(min(distance_center))
        iou = cal_iou(gt_object, np_centernet_bev[min_index])
        is_true = (iou >= IOU_THRESHOLD)
        gtp1 += (is_true == True)
        fn1 += (is_true == False)

    for gt_object in np_gt:
        diff_center = (gt_object - np_refine_pred)[:, :2]  # nx2
        distance_center = [np.sqrt(x * x + y * y) for (x, y) in diff_center]  # n
        min_index = distance_center.index(min(distance_center))
        iou = cal_iou(gt_object, np_refine_pred[min_index])
        is_true = (iou >= IOU_THRESHOLD)
        gtp2 += (is_true == True)
        fn2 += (is_true == False)

    return tp, fp, tp2, fp2, fn1, fn2,gtp1,gtp2

def infer_one_img(detector, pre_processed_images, ind, K, D, new_K, bTc, ex4, refine_3d_model):
    one_img_loss_CenterNetBev = 0
    one_img_objects_num = 0
    ret, vehicle_feature_map, vehicle_wheel_points, vehicle_scores = detector.run(pre_processed_images,
                                                                                  img_id=ind)
    # vehicle_feature_map nx64 Tensor           vehicle_wheel_points nx8 ndarray        vehicle_scores nx1 ndarray
    if vehicle_scores.shape[0] == 0:
        return None
    threshold_mask = (vehicle_scores.reshape(-1) >= OBJECT_THRESHOLD)
    threshold_indices = np.nonzero(threshold_mask)[0]
    vehicle_feature_map = vehicle_feature_map[threshold_indices, :]
    vehicle_wheel_points = vehicle_wheel_points[threshold_indices, :]
    if len(threshold_indices) == 0:
        return None

    vehicle_wheel_points = vehicle_wheel_points.reshape((-1, 4, 1, 2))
    vehicle_wheel_points = [x for x in vehicle_wheel_points]

    undistorted_oneImgObjects = undistort_contours(vehicle_wheel_points, K, D, new_K)
    oneImagePts3d = IPM_contours(undistorted_oneImgObjects, new_K, bTc, ex4,
                                 p=[0, 0, 0, 0.332, 0])  # oneImagePts3d: pandar激光雷达坐标系

    one_img_objects = torch.zeros(size=(len(oneImagePts3d), 5), device='cuda')
    for object_index, oneObject in enumerate(oneImagePts3d):
        keypoints = np.ndarray([4, 2], dtype=float)
        for pts_index in range(4):
            keypoints[pts_index][0] = oneObject[pts_index][0]  # -25 to 25
            keypoints[pts_index][1] = oneObject[pts_index][1]  # -50 to 0
        # keypoints x范围 -8到8 右为正方向。 y范围：0到16 上为正方向

        # 估计矩形的初始形状：
        center_x = np.mean(keypoints[:, 0])
        center_y = np.mean(keypoints[:, 1])
        pts_1 = keypoints - np.array([center_x, center_y])
        back_center = (pts_1[2] + pts_1[3]) / 2
        if abs(back_center[0]) < 1e-7:  # ziji todo ttt
            back_center[0] = 1e-7
        rotation = np.arctan(back_center[1] / back_center[0]) + (np.pi / 2)  # rotation:0,pi
        if back_center[0] < 0:
            rotation -= np.pi
        # rotation: -pi to pi
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)  # 顺时针旋转矩阵
        pts_2 = np.matmul(R, pts_1.T).T
        l = (pts_2[0][1] + pts_2[1][1] - pts_2[2][1] - pts_2[3][1]) / 2
        w = (pts_2[1][0] + pts_2[2][0] - pts_2[0][0] - pts_2[3][0]) / 2
        if l < 0:
            l = -l
        if w < 0:
            w = -w

        # 迭代优化：
        x0 = np.array([center_x, center_y, l, w, rotation], dtype=np.float)
        res = minimize(bev_bbox_error, x0, args=keypoints, method='nelder-mead',
                       options={'disp': False})
        # rectangle_final = compute_box_bev(res.x)

        # BGR
        # paint_bev(im_bev, keypoints, (255, 0, 0))
        # paint_bev(im_bev, rectangle_final, (0, 0, 255))
        # objects_bev_pred = np.concatenate((objects_bev_pred, rectangle_final.reshape(1, 4, 2)))

        one_img_objects[object_index] = torch.Tensor(res.x).cuda()
        one_img_objects[object_index][4] = trucate_angle(one_img_objects[object_index][4])
    # one_img_objects Tensor: nx5
    # one_img_ipm = torch.zeros(size=(len(oneImagePts3d), 8), device='cuda')
    one_img_centers = torch.zeros(size=(len(oneImagePts3d), 2), device='cuda')
    for i in range(len(oneImagePts3d)):
        for j in range(4):
            # one_img_ipm[i, j*2] = oneImagePts3d[i][j][0]
            # one_img_ipm[i, j*2+1] = oneImagePts3d[i][j][1]
            one_img_centers[i, 0] += oneImagePts3d[i][j][0]
            one_img_centers[i, 1] += oneImagePts3d[i][j][1]
    # one_img_ipm: shape nx8 Tensor
    one_img_centers = one_img_centers / 4  # nx2

    # one_img_ipm_featuremap = torch.cat((one_img_ipm, vehicle_feature_map), dim=1) # nx72
    # pred = refine_3d_model(one_img_ipm_featuremap) # nx8
    pred = refine_3d_model(one_img_objects)  # nx5
    # x,y,l,w,rotation
    one_img_gt = pre_processed_images['gt_tensor'][0].cuda()
    one_img_loss = torch.zeros(1, requires_grad=True).cuda()
    for one_object_pred_index in range(pred.shape[0]):
        # pred[one_object_pred_index,4] = trucate_angle(pred[one_object_pred_index,4])
        min_distance = 99999999
        min_index = None
        for one_object_gt_index in range(one_img_gt.shape[0]):
            cur_dis_2 = (one_img_centers[one_object_pred_index, 0] - one_img_gt[
                one_object_gt_index, 0]) ** 2 \
                        + (one_img_centers[one_object_pred_index, 1] - one_img_gt[
                one_object_gt_index, 1]) ** 2
            if cur_dis_2 < min_distance:
                min_index = one_object_gt_index
                min_distance = cur_dis_2
        # cal loss
        if min_distance > (detector.opt.match_max_center_dis ** 2):  # ziji todo
            continue
        if min_index != None:
            if detector.opt.refine_loss == 'mse':
                diff = nn.MSELoss(reduction='none')(pred[one_object_pred_index][0:4], one_img_gt[min_index][0:4])
                one_object_loss = diff.sum()
                angle_diff = torch.abs(pred[one_object_pred_index, 4] - one_img_gt[min_index, 4])  # 0,2pi
                angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff) ** 2

                one_object_loss += angle_diff * 10
                one_img_loss += one_object_loss

                diff2 = nn.MSELoss(reduction='none')(one_img_objects[one_object_pred_index][0:4],
                                                     one_img_gt[min_index][0:4])
                one_object_loss2 = diff2.sum()
                angle_diff2 = torch.abs(one_img_objects[one_object_pred_index, 4] - one_img_gt[min_index, 4])
                angle_diff2 = torch.min(angle_diff2, 2 * np.pi - angle_diff2) ** 2

                one_object_loss2 += angle_diff2 * 10
                one_img_loss_CenterNetBev += one_object_loss2.item()
            elif detector.opt.refine_loss == 'l1':
                diff = nn.L1Loss(reduction='none')(pred[one_object_pred_index][0:4], one_img_gt[min_index][0:4])
                one_object_loss = diff.sum()
                angle_diff = torch.abs(pred[one_object_pred_index, 4] - one_img_gt[min_index, 4])  # 0,2pi
                angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)

                one_object_loss += angle_diff * 3
                one_img_loss += one_object_loss

                diff2 = nn.L1Loss(reduction='none')(one_img_objects[one_object_pred_index][0:4],
                                                     one_img_gt[min_index][0:4])
                one_object_loss2 = diff2.sum()
                angle_diff2 = torch.abs(one_img_objects[one_object_pred_index, 4] - one_img_gt[min_index, 4])
                angle_diff2 = torch.min(angle_diff2, 2 * np.pi - angle_diff2)

                one_object_loss2 += angle_diff2 * 3
                one_img_loss_CenterNetBev += one_object_loss2.item()

            else:
                raise NotImplementedError()

            print('one_object_loss', one_object_loss.item(), 'one_object_loss2',
                  one_object_loss2.item())
            one_img_objects_num += 1

    return one_img_loss, one_img_loss_CenterNetBev, one_img_objects_num, pred, one_img_objects, one_img_gt, ret
