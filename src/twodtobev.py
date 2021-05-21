#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
DISTANCE_TRUE = 2.0
OBJECT_THRESHOLD = 0.3




import time
import numpy as np
import cv2
import yaml
import pyquaternion
import math
import json
from scipy.optimize import minimize

np.set_printoptions(suppress=True)

def is_freespace(contour_index, hierarchy):
    #print (hierarchy)
    #hierarchy = hierarchy[0]
    levels = 0
    current_contour_index = contour_index
    for i in hierarchy :  # only for limiting the max loops count.
        parent = hierarchy[current_contour_index]
        #print(parent)
        if parent[3] == -1: # root
            break
        else:
            current_contour_index = parent[3]
            levels += 1
    
    return levels % 2 == 0
    

# return exterior and interior
def extract_contours(image): 
    ret, binary = cv2.threshold(image,0,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)  
    return contours , None if hierarchy is None else hierarchy[0]


def load_cam_param():
    with open(os.path.join(calib_path,"front_center.yaml")) as f:
        intrinsic = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(calib_path,"FRONT_CENTER_CAMERA.yaml")) as f:
        extrinsic = yaml.load(f, Loader=yaml.FullLoader)
    return intrinsic , extrinsic 
    

def cam_intrinsic(intrinsic):
    cv_d = np.array([intrinsic['k1'], \
                     intrinsic['k2'], \
                     intrinsic['k3'], \
                     intrinsic['k4'], ]  ,dtype=np.float64)
    cv_k = np.eye(3,dtype=np.float64)
    cv_k[0][0] = intrinsic['fx']
    cv_k[1][1] = intrinsic['fy']
    cv_k[0][2] = intrinsic['cx']
    cv_k[1][2] = intrinsic['cy']
    cv_k[0][1] = intrinsic['skew']
    cv_k = np.matrix(cv_k)
    image_size = (intrinsic['width'] ,  intrinsic['height'])
    
    #new_K ,roi = cv2.getOptimalNewCameraMatrix(cv_k, cv_d, image_size, 0);
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cv_k, cv_d, image_size, R = np.eye(3))

    return cv_k ,cv_d, new_K

def cam_extrinsic(extrinsic):
    pos = np.array([extrinsic['x'],extrinsic['y'],extrinsic['z'] ])
    quat = pyquaternion.Quaternion( extrinsic['qw'],extrinsic['qx'],extrinsic['qy'], extrinsic['qz'] )
    
    
    bTc = np.eye(4, dtype=np.float64 )
    bTc[:3,:3] = quat.rotation_matrix
    bTc[0,3] = pos[0]
    bTc[1,3] = pos[1]
    bTc[2,3] = pos[2]
    
    return pos , quat , np.matrix(bTc)
    

def undistort_contours( contours , K , D , new_K ):
    
    pn = []
    for c in range(len(contours)):
        #print (c, len(contours), contours[c])
        pn.append ( len(contours[c]) )

    flatten_points = []
    for c in range(len(contours)):
        for e in range(pn[c]):
            flatten_points.append(contours[c][e][0])
    flatten_points = np.array(flatten_points).reshape((-1,1,2)).astype(np.float32)
    
    #print (flatten_points)
    
    undistort_points = cv2.fisheye.undistortPoints(flatten_points, K, D, R = np.eye(3), P = new_K)

    #restore
    undistorted_contours = contours #restore to (CN, PN, 1, 2)
    flatten_idx = 0
    for c in range(len(contours)):
        for e in range(pn[c]):
            undistorted_contours[c][e][0][0] = undistort_points[flatten_idx].tolist()[0][0]
            undistorted_contours[c][e][0][1] = undistort_points[flatten_idx].tolist()[0][1]
            flatten_idx += 1
    #print(undistort_contours)
    return undistorted_contours
    
    
def undistort_image(image, K, D, new_K ):
    #print (image.shape)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, R = np.eye(3), P = new_K,size=  (1920,1020) , m1type = cv2.CV_32FC1  )
    undistorted_image = cv2.remap(image, map_x, map_y,  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    #undistorted_image = cv2.fisheye.undistort(image, K, D,None,  K)
    return undistorted_image
    



def restore_contours_from_points(contours, points):
    
    points = points[:,:2]
    pn = []
    for c in range(len(contours)):
        #print (c, len(contours), contours[c])
        pn.append ( len(contours[c]) )

    flatten_points = []
    for c in range(len(contours)):
        for e in range(pn[c]):
            flatten_points.append(contours[c][e][0])
    flatten_points = np.array(flatten_points).reshape((-1,1,2)).astype(np.float32)
    
    undistort_points = points.reshape((-1,1,2)).astype(np.float32) #cv2.fisheye.undistortPoints(flatten_points, K, D, R = np.eye(3), P = new_K)

    #restore 
    undistorted_contours = contours #restore to (CN, PN, 1, 2)
    flatten_idx = 0
    for c in range(len(contours)):
        for e in range(pn[c]):
            undistorted_contours[c][e][0][0] = undistort_points[flatten_idx].tolist()[0][0]
            undistorted_contours[c][e][0][1] = undistort_points[flatten_idx].tolist()[0][1]
            flatten_idx += 1
    #print(undistort_contours)
    return undistorted_contours
    

def IPM_contours(contours, K, bTc, ex4, p=[0.00127438, 0.00339385, 0.00429253, 0.35888672, 1.48804348]):
    if (len(contours) < 1):
        return None
    pn = []
    for c in range(len(contours)):
        # print (c, len(contours), contours[c])
        pn.append(len(contours[c]))

    flatten_points = []
    for c in range(len(contours)):
        for e in range(pn[c]):
            flatten_points.append(contours[c][e][0])
    flatten_points = np.array(flatten_points).reshape((-1, 2)).astype(np.float32)
    flatten_points = np.concatenate((flatten_points, [[1] for x in flatten_points]), axis=1)
    ####
    # to cam
    K_inv = np.matrix(K).I
    inv_vec = K_inv * flatten_points.T
    # convert from uv to imu coord , no need
    inv_vec_cam = np.array(inv_vec.T)
    inv_vec_cam = inv_vec_cam / np.linalg.norm(inv_vec_cam, axis=1, keepdims=True)

    # inv_vec_body_rot = (Quat.rotation_matrix * np.matrix(inv_vec_cam).T).T
    # inv_vec_body_rot = (Quat.rotation_matrix * np.matrix(inv_vec_cam).T).T
    # inv_vec_body_rot -= Pos

    p_roll = p[0]
    roll_matrix = np.matrix([[1, 0, 0],
                             [0, math.cos(p_roll), -math.sin(p_roll)],
                             [0, math.sin(p_roll), math.cos(p_roll)]])

    p_pitch = p[1]
    pitch_matrix = np.matrix([[math.cos(p_pitch), 0, math.sin(p_pitch)],
                              [0, 1, 0],
                              [-math.sin(p_pitch), 0, math.cos(p_pitch)]])

    p_yaw = p[2]
    yaw_matrix = np.matrix([[math.cos(p_yaw), -math.sin(p_yaw), 0],
                            [math.sin(p_yaw), math.cos(p_yaw), 0],
                            [0, 0, 1]])

    # rot_matrix = roll_matrix * pitch_matrix * yaw_matrix
    rot_matrix = yaw_matrix * pitch_matrix * roll_matrix

    gTb = np.matrix(np.eye(4, dtype=np.float64))
    gTb[:3, :3] = rot_matrix
    gTb[2, 3] = p[3]

    gTc = gTb * bTc

    inv_vec_body_rot = (gTc[:3, :3] * np.matrix(inv_vec_cam).T).T

    cam_above_ground = gTc[2, 3]  # p[3] + Pos[2]

    print("estimate cam above ground : " + str(cam_above_ground))
    d_length = -cam_above_ground / inv_vec_body_rot[:, 2]
    point_3d = np.array(inv_vec_body_rot) * np.array(d_length).reshape(-1, 1)  # - Pos
    # point_3d -= Pos
    point_3d[:, 0:2] += np.array(gTc[0:2, 3]).flatten()



    # todo ziji: 蒙：
    point_3d[:,2] = 0
    # point_3d 现在是ground坐标系

    point_ground_homo = np.concatenate((point_3d.T, np.ones((1, point_3d.shape[0]))))

    point_camera_homo = (np.matmul(gTc.I, point_ground_homo))





    #todo ziji 验证代码 下面两行可以删除 删不删都行
    point_camera_2d_homo = np.matmul(K, point_camera_homo[:3, :]).T
    point_camera_2d = point_camera_2d_homo[:, :2] / point_camera_2d_homo[:, 2:3]
    # 此处验证与coutours中的原2D接地点坐标相同，所以上面point_3d肯定是地面坐标系的坐标。




    point_lidar_homo = (np.matmul(ex4.I, point_camera_homo)).T
    point_lidar_homo = point_lidar_homo.A
    point3d_lidar = point_lidar_homo[:, :3] / point_lidar_homo[:, 3:4]

    # reshape
    point_3d_reshaped = []
    pt_index = 0
    for c in range(len(contours)):
        point_3d_reshaped.append([])
        for e in range(pn[c]):
            point_3d_reshaped[c].append(point3d_lidar[pt_index])
            pt_index += 1

    return point_3d_reshaped

def compute_box_bev(x):
    # return shape (4,2)
    location = x[:2]
    dim = x[2:4]
    rotation = x[4]  # 正数：左转。负数：右转


    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]], dtype=np.float32) # 逆时针旋转矩阵

    l, w = dim[0], dim[1]

    x_corners = [-w/2, w/2, w/2, -w/2]
    z_corners = [l / 2, l / 2, -l / 2, -l / 2]

    corners = np.array([x_corners, z_corners], dtype=np.float32)
    corners_bev = np.matmul(R, corners)
    corners_bev = corners_bev + np.array(location, dtype=np.float32).reshape(2, 1)
    return corners_bev.transpose(1, 0)



def trucate_angle(alpha):
    while alpha > math.pi or alpha < -math.pi:
        if alpha > math.pi:
            alpha -= 2*math.pi
        if alpha < -math.pi:
            alpha += 2 * math.pi
    return alpha

def Gen3DBox(x, y, z, w, h, l, alpha):
    """
    Y point back, X point left
    w -> Y, h -> X, l -> Z
    """
    t = np.array([x, y, z], np.float32)
    pts = [
        [w/2, -l/2, h/2],
        [-w/2, -l/2, h/2],
        [-w/2, -l/2, -h/2],
        [w/2, -l/2, -h/2],
        [w/2, l/2, h/2],
        [-w/2, l/2, h/2],
        [-w/2, l/2, -h/2],
        [w/2, l/2, -h/2],
        [0, 0, 0]
    ]
    pts = np.array(pts, np.float32)
    R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                  [np.sin(alpha), np.cos(alpha),  0],
                  [0,          0,  1]], np.float32)
    pts = R.dot(pts.T).T + t
    return pts

def cal_tp_fp_fn_one_image(objects_bev_pred, objects_bev_gt):
    # objects_bev_pred和objects_bev_gt 都是nx4x2
    centers_pred = np.mean(objects_bev_pred, axis=1) # centers_pred:nx2:每个物体的中心点
    centers_gt = np.mean(objects_bev_gt,axis=1) # centers_gt:nx2:每个物体的中心点

    tp, fp, tp2, fn = 0,0,0,0  # tp和tp2一样，一个意思
    # 计算tp fp
    for one_center_pred in centers_pred:
        diff_center = centers_gt-one_center_pred # diff_center: nx2     每个gt物体中心点与one_center_pred差距
        distance_center = [np.sqrt(x*x+y*y) for (x,y) in diff_center] # distance_center:nx1
        dis_true = np.array([ x < DISTANCE_TRUE for x in distance_center]) # nx1
        true_cnt = np.sum(dis_true)
        is_true = (true_cnt >=1)
        tp += (is_true == True)
        fp += (is_true == False)

    # 计算tp, fn
    for one_center_gt in centers_gt:
        diff_center = centers_pred-one_center_gt # diff_center: nx2     每个gt物体中心点与one_center_pred差距
        distance_center = [np.sqrt(x*x+y*y) for (x,y) in diff_center] # distance_center:nx1
        dis_true = np.array([ x < DISTANCE_TRUE for x in distance_center]) # nx1
        true_cnt = np.sum(dis_true)
        is_true = (true_cnt >=1)
        tp2 += (is_true == True)
        fn += (is_true == False)


    return tp, fp, fn





def treat_one_img(objects, file_name, file_path='1218_03_anno/'):
    # objects： 物体数x4x3
    def bev_bbox_error(x, keypoints):
        rectangle = compute_box_bev(x)
        rec_dif = rectangle - keypoints
        rec_dif_2 = rec_dif ** 2
        error = np.sum(rec_dif_2)
        return error


    def paint_bev(im_bev, points_bev, lineColor3d, width = 1000, height = 1000):
        points = np.copy(points_bev)
        points[:,0] = (25 - points[:,0]) * (width/50)
        points[:,1] = (50 + points[:,1]) * (height / 50)

        points = points.astype(np.int)
        cv2.line(im_bev, (points[0][0], points[0][1]), (points[1][0], points[1][1]), lineColor3d, 1)
        cv2.line(im_bev, (points[1][0], points[1][1]), (points[2][0], points[2][1]), lineColor3d, 1)
        cv2.line(im_bev, (points[2][0], points[2][1]), (points[3][0], points[3][1]), lineColor3d, 1)
        cv2.line(im_bev, (points[0][0], points[0][1]), (points[3][0], points[3][1]), lineColor3d, 1)
    
    im_bev = np.ones([1000, 1000], dtype=np.uint8)
    im_bev = cv2.cvtColor(im_bev, cv2.COLOR_GRAY2RGB)
    im_bev *= 255

    objects_bev_pred = np.ndarray([0, 4, 2])
    objects_bev_gt = np.ndarray([0, 4, 2])

    for oneObject in objects:
        if len(oneObject)!= 4:   #todo ziji!!!!!
            continue

        keypoints = np.ndarray([4,2],dtype=float)
        for pts_index in range(4):
            keypoints[pts_index][0] = oneObject[pts_index][0]  # -25 to 25
            keypoints[pts_index][1] = oneObject[pts_index][1]  # -50 to 0
        # keypoints x范围 -8到8 右为正方向。 y范围：0到16 上为正方向


        # 估计矩形的初始形状：
        center_x = np.mean(keypoints[:,0])
        center_y = np.mean(keypoints[:,1])
        pts_1 = keypoints - np.array([center_x,center_y])
        back_center = (pts_1[2] + pts_1[3]) / 2
        if abs(back_center[0]) < 1e-7: # ziji todo ttt
            back_center[0] = 1e-7
        rotation = np.arctan(back_center[1]/back_center[0]) + (np.pi / 2)
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)  # 顺时针旋转矩阵
        pts_2 = np.matmul(R, pts_1.T).T
        l = (pts_2[0][1] + pts_2[1][1] - pts_2[2][1]-pts_2[3][1]) / 2
        w = (pts_2[1][0] + pts_2[2][0] - pts_2[0][0] - pts_2[3][0]) / 2

        # 迭代优化：
        x0 = np.array([center_x, center_y, l, w, rotation], dtype=np.float)
        res = minimize(bev_bbox_error, x0, args=keypoints, method='nelder-mead', options={'disp': False})
        rectangle_final = compute_box_bev(res.x)

        # BGR
        paint_bev(im_bev, keypoints, (255, 0, 0))
        paint_bev(im_bev, rectangle_final,(0,0,255))
        objects_bev_pred = np.concatenate((objects_bev_pred, rectangle_final.reshape(1,4,2)))







    # 绘制ground truth
    try:
        j = json.load(open(file_path + file_name+'.json', 'r'))[0]
    except FileNotFoundError:
        print("无gt文件")
        return im_bev, (0, 0, 0)

    if u'\u6709\u70b9\u4e91\u6846' not in j.keys():
        print("无点云框")
        return im_bev, (0,0,0)

    for obs in j[u'\u6709\u70b9\u4e91\u6846']:
        center = obs['center']
        x, y, z = center['x'], center['y'], center['z']
        l, w, h = obs['width'], obs['height'], obs['depth']
        alpha = trucate_angle(obs['rotation']['z'] - np.pi / 2)
        cls = obs['attributes'][u'\u5c5e\u6027'][0]
        pts_gt = Gen3DBox(x, y, z, w, h, l, alpha)
        need_pts_gt = pts_gt[[2,3,7,6],:2]
        paint_bev(im_bev, need_pts_gt, (0, 255,0))
        objects_bev_gt = np.concatenate((objects_bev_gt, need_pts_gt.reshape(1,4,2)))


    # 使用objects_bev_pred和object_bev_gt计算精度召回率
    one_img_tp, one_img_fp, one_img_fn = cal_tp_fp_fn_one_image(objects_bev_pred, objects_bev_gt)

    return im_bev, (one_img_tp, one_img_fp, one_img_fn)


def main(object_threshold = OBJECT_THRESHOLD):
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

    bTc = ex3 * ex2 * ex1.I # todo ziji


    results_file_name = "results_test2_dla_3x"
    results_file = open(results_file_name + ".json",'r')
    object_dict = json.load(results_file)

    file = open("heduo_5cls_keypoints_test2.json", 'r')
    img_dict = json.load(file)['images']

    for oneImg in img_dict:
        oneImg['objects'] = []

    img_index = 0
    for oneObject in object_dict:
        image_id = oneObject['image_id']
        while img_dict[img_index]['id'] != image_id:
            img_index += 1
            if img_index >= len(img_dict):
                raise ValueError('the order of results.json has error!!!')

        img_dict[img_index]['objects'].append(oneObject)


    img_index = 0
    all_tp, all_fp, all_fn = 0,0,0
    # allPts3d = []
    for oneImg in img_dict:
        oneImgObjects = []
        for oneObject in oneImg['objects']:
            if oneObject['score'] < object_threshold:
                continue
            keypoints = np.array(oneObject['keypoints'])
            keypoints = keypoints.reshape([-1,3])
            keypoints = filter(lambda x: x[2]==1.0, keypoints) # todo ziji 0511看 应该改成 x[2]!=0.0
            keypoints = [x for x in keypoints]
            if len(keypoints) == 0:
                continue
            pts = [x[0:2] for x in keypoints]
            pts = np.array(pts)
            pts = pts.reshape([-1, 1, 2]) #pts:4x1x2
            oneImgObjects.append(pts)

        if len(oneImgObjects) == 0:
            img_index += 1
            continue
        # oneImgObjects: nx4x1x2
        undistorted_oneImgObjects = undistort_contours(oneImgObjects, K, D, new_K)
        oneImagePts3d = IPM_contours(undistorted_oneImgObjects, new_K, bTc, ex4, p=[0, 0, 0, 0.332, 0]) #oneImagePts3d: pandar激光雷达坐标系

        # allPts3d.append(oneImagePts3d)
        bev_img, (one_img_tp, one_img_fp, one_img_fn) = treat_one_img(oneImagePts3d, oneImg['file_name'])
        cv2.imwrite(os.path.join('bev_gt', results_file_name, "{0:03d}__".format(img_index) + oneImg['file_name']), bev_img) #todo ziji ttt

        all_tp += one_img_tp
        all_fp += one_img_fp
        all_fn += one_img_fn

        print("{0:03d}__".format(img_index) + oneImg['file_name'], "\ttp:%d, fp:%d, fn:%d" % (one_img_tp, one_img_fp, one_img_fn))

        img_index += 1
        if img_index >= 1000:
            break

    precision = all_tp / (all_tp+all_fp)
    recall = all_tp / (all_tp+all_fn)

    print(results_file_name, "precision:%f, recall:%f" % (precision, recall))

if __name__ == "__main__":
    main()
    
    

