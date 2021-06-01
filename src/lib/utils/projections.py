import os
import sys
import json
from pyquaternion import Quaternion
import cv2
import numpy as np
import math


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


def trucate_angle(alpha):
    while alpha > math.pi or alpha < -math.pi:
        if alpha > math.pi:
            alpha -= 2*math.pi
        if alpha < -math.pi:
            alpha += 2 * math.pi
    return alpha


def LoadCameraParamsRaw(camera_param_file):
    fpath = camera_param_file
    j = json.load(open(fpath, 'r'))[0]
    position = j['position']
    heading = j['heading']
    skew = j['skew']
    fx, fy, cx, cy = j['fx'], j['fy'], j['cx'], j['cy']
    k1, k2, k3, k4 = j['k1'], j['k2'], j['k3'], j['k4']
    p1, p2 = j['p1'], j['p2']
    return k1, k2, k3, k4, p1, p2, fx, fy, cx, cy, skew, heading, position

def DistortPointsFisheye(x, y, z, k1, k2, k3, k4, fx, fy, cx, cy, alpha=0):
    a = x / z
    b = y / z
    r2 = a**2 + b**2
    r = math.sqrt(r2)
    theta = math.atan(r)

    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta2 * theta4
    theta8 = theta4 ** 2

    theta_d = theta * (1.0 + k1 * theta2 + k2 *
                       theta4 + k3 * theta6 + k4 * theta8)
    if r > 1e-8:
        scale = 1.0 / r
    else:
        scale = 1.0

    u_ = (theta_d * scale) * a
    v_ = (theta_d * scale) * b

    u_ = fx * (u_ + alpha * v_) + cx
    v_ = fy * v_ + cy

    return u_, v_

def Project3DToCamera(camera, vector):
    k1, k2, k3, k4, p1, p2, fx, fy, cx, cy, skew, heading, position = camera
    cam_heading = Quaternion(
        heading['w'], heading['x'], heading['y'], heading['z']).rotation_matrix
    cam_position = np.array(
        [position['x'], position['y'], position['z']], np.float32).reshape(3, 1)

    point_in_cam = vector - cam_position
    point_in_cam = np.linalg.inv(cam_heading).dot(point_in_cam)

    if point_in_cam[2] <= 0:
        return None

    correction = (k1 is not None) and (k2 is not None) and (
        p1 is not None) and (p2 is not None)

    if correction:
        if p1 != 0 or p2 != 0:
            print('Not Impl')
            pass
        else:
            return DistortPointsFisheye(point_in_cam[0], point_in_cam[1], point_in_cam[2], k1, k2, k3, k4, fx, fy, cx, cy, skew)
    else:
        return point_in_cam[0] / point_in_cam[2] * fx + cx, point_in_cam[1] / point_in_cam[2] * fy + cy


def project(pts, camera):
    ret = []
    is_vis = np.ones((9,1))

    for i, pt in enumerate(pts):
        aaa = Project3DToCamera(camera, pt.reshape(3, 1))
        if aaa is None:
            is_vis[i] = 0
            ret.append((np.nan, np.nan))
        else:
            ret.append(aaa)
    ret = np.array(ret, dtype=np.float32).reshape(9,2)
    vis_num = is_vis.sum()
    is_vis = is_vis*2
    pts_2d = np.column_stack((ret,is_vis))
    return pts_2d, vis_num

if __name__ == 'main':
    project()