from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # todo

import json
import cv2
import yaml
import pyquaternion
import math
import json
from itertools import chain
from scipy.optimize import minimize
import numpy as np
import time
from progress.bar import Bar
import torch
from torch import nn
from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory, get_dataset
from detectors.detector_factory import detector_factory
from twodtobev import undistort_contours, IPM_contours, cam_intrinsic, cam_extrinsic
from refine_3d_network import Refine_3d_Network
from models.model import save_model, load_model
from visdom import Visdom
from torch.optim.lr_scheduler import CosineAnnealingLR
OBJECT_THRESHOLD = 0.0

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
        gt_tensor = torch.zeros(size=(len(gt_ydyk), 6),requires_grad=False)
        for i in range(len(gt_ydyk)):
            gt_tensor[i][0] = gt_ydyk[i]['center']['x']
            gt_tensor[i][1] = gt_ydyk[i]['center']['y']
            gt_tensor[i][2] = gt_ydyk[i]['width']
            gt_tensor[i][3] = gt_ydyk[i]['depth']
            gt_tensor[i][4] = np.cos(gt_ydyk[i]['rotation']['z'])
            gt_tensor[i][5] = np.sin(gt_ydyk[i]['rotation']['z'])

        img_relative_path = '/'.join(anno_json['img'].split('/')[-3:])
        img_path = os.path.join(self.img_dir, img_relative_path)
        image = cv2.imread(img_path)
        images, meta = {}, {}

        assert opt.test_scales == [1.0]
        for scale in opt.test_scales:
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

def prefetch_test(opt):
    K, D, new_K, bTc, ex4 = load_camera_parameter()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    split = opt.split
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]
    opt.save_infer_dir = os.path.join(opt.save_dir, split)
    detector = Detector(opt)

    #Dataset = dataset_factory[opt.dataset]
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    # Dataset = get_dataset(opt.dataset, opt.task)
    # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    dataset = Heduo_2nd_batch_Dataset(opt, detector.pre_process, opt.train_anno_dir)
    dataset_val = Heduo_2nd_batch_Dataset(opt, detector.pre_process, opt.val_anno_dir)

    # data_loader = torch.utils.data.DataLoader(
    #     PrefetchDataset(opt, dataset, detector.pre_process, gt=(opt.debug == 2)),
    #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    refine_3d_model = Refine_3d_Network(72,6)
    load_model(refine_3d_model, opt.refine_model_dir)
    refine_3d_model = refine_3d_model.cuda()
    #optimizer = torch.optim.Adam(chain(detector.model.parameters(), refine_3d_model.parameters()), lr=1e-4)
    optimizer = torch.optim.Adam(refine_3d_model.parameters(), lr=1e-4)
    scheduler_1 = CosineAnnealingLR(optimizer, T_max=40)



    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    val_loss_min = 1e30


    with torch.no_grad():
        val_loss_total = 0
        for ind, (img_id, pre_processed_images) in enumerate(data_loader_val):
            ret, vehicle_feature_map, vehicle_wheel_points, vehicle_scores = detector.run(pre_processed_images,
                                                                                          img_id=ind)
            # vehicle_feature_map nx64 Tensor           vehicle_wheel_points nx8 ndarray        vehicle_scores nx1 ndarray
            if vehicle_scores.shape[0] == 0:
                continue
            threshold_mask = (vehicle_scores.reshape(-1) >= OBJECT_THRESHOLD)
            threshold_indices = np.nonzero(threshold_mask)[0]
            vehicle_feature_map = vehicle_feature_map[threshold_indices, :]
            vehicle_wheel_points = vehicle_wheel_points[threshold_indices, :]
            if len(threshold_indices) == 0:
                continue



            vehicle_wheel_points = vehicle_wheel_points.reshape((-1, 4, 1, 2))
            vehicle_wheel_points = [x for x in vehicle_wheel_points]

            undistorted_oneImgObjects = undistort_contours(vehicle_wheel_points, K, D, new_K)
            oneImagePts3d = IPM_contours(undistorted_oneImgObjects, new_K, bTc, ex4,
                                         p=[0, 0, 0, 0.332, 0])  # oneImagePts3d: pandar激光雷达坐标系
            one_img_ipm = torch.zeros(size=(len(oneImagePts3d), 8), device='cuda')
            one_img_centers = torch.zeros(size=(len(oneImagePts3d), 2), device='cuda')
            for i in range(len(oneImagePts3d)):
                for j in range(4):
                    one_img_ipm[i, j * 2] = oneImagePts3d[i][j][0]
                    one_img_ipm[i, j * 2 + 1] = oneImagePts3d[i][j][1]
                    one_img_centers[i, 0] += oneImagePts3d[i][j][0]
                    one_img_centers[i, 1] += oneImagePts3d[i][j][1]
            # one_img_ipm: shape nx8 Tensor
            one_img_centers = one_img_centers / 4  # nx2

            one_img_ipm_featuremap = torch.cat((one_img_ipm, vehicle_feature_map), dim=1)  # nx72
            pred = refine_3d_model(one_img_ipm_featuremap)  # nx8

            one_img_gt = pre_processed_images['gt_tensor'][0].cuda()
            one_img_loss = torch.zeros(1, requires_grad=True).cuda()
            for one_object_pred_index in range(pred.shape[0]):
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
                if min_index != None:
                    diff = nn.MSELoss(reduction='none')(pred[one_object_pred_index], one_img_gt[min_index])
                    diff[4] *= 10
                    diff[5] *= 10
                    one_object_loss = diff.sum()
                    one_img_loss += one_object_loss

            val_loss_total += one_img_loss.detach().item()
            print("val_one_img_loss:", one_img_loss.item())

            # results[img_id.numpy().astype(np.int32)[0]] = ret['results']
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
                Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                    t, tm=avg_time_stats[t])
            bar.next()
        val_loss_total /= len(dataset_val)
        print("val_loss_total: ",val_loss_total)



    bar.finish()
    # if opt.debug <= 1:
    #     dataset.save_results(results, opt.save_dir)
    # else:
    #     dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().init()
    prefetch_test(opt)