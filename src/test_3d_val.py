from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import yaml
import pyquaternion
import math
import json
from itertools import chain
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

from refine_3d_easy_network import Refine_3d_easy_Network
from models.model import save_model, load_model
from visdom import Visdom
from torch.optim.lr_scheduler import CosineAnnealingLR
from util_3d import *

def paint_bev(im_bev, points_bev, lineColor3d, width=1000, height=1000, thick=4):
    points = np.copy(points_bev)
    points[:, 0] = (25 - points[:, 0]) * (width / 50)
    points[:, 1] = (50 + points[:, 1]) * (height / 50)

    points = points.astype(np.int)
    cv2.line(im_bev, (points[0][0], points[0][1]), (points[1][0], points[1][1]), lineColor3d, thick)
    cv2.line(im_bev, (points[1][0], points[1][1]), (points[2][0], points[2][1]), lineColor3d, thick)
    cv2.line(im_bev, (points[2][0], points[2][1]), (points[3][0], points[3][1]), lineColor3d, thick)
    cv2.line(im_bev, (points[0][0], points[0][1]), (points[3][0], points[3][1]), lineColor3d, thick)


def paint_bev_all(np_centernet_bev,np_refine_pred,np_gt):
    # 输入都是nx5


    im_bev = np.ones([1000, 1000], dtype=np.uint8)
    im_bev = cv2.cvtColor(im_bev, cv2.COLOR_GRAY2RGB)
    im_bev *= 255
    #BGR

    for centernet_bev_object in np_centernet_bev:
        centernet_bev_pts = compute_box_bev(centernet_bev_object)
        paint_bev(im_bev, centernet_bev_pts, (0, 0, 255))

    for pred_object in np_refine_pred:
        pred_pts = compute_box_bev(pred_object)
        paint_bev(im_bev, pred_pts, (0, 140, 255))

    for gt_object in np_gt:
        gt_pts = compute_box_bev(gt_object)
        paint_bev(im_bev, gt_pts, (0, 255, 0))

    return im_bev
def paint_bev_nogt(np_refine_pred, image):
    # 输入都是nx5

    im_bev = np.ones([510, 510], dtype=np.uint8)
    im_bev = cv2.cvtColor(im_bev, cv2.COLOR_GRAY2RGB)
    im_bev *= 255
    # BGR
    for pred_object in np_refine_pred:
        pred_pts = compute_box_bev(pred_object)
        paint_bev(im_bev, pred_pts, (0, 140, 255), width=510, height=510, thick=3)

    image_resize = cv2.resize(image,(960,510))
    image_cat = np.concatenate((image_resize, im_bev), axis=1)
    return image_cat

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

    if opt.img_nogt_dir != None:
        dataset_val = Heduo_2nd_batch_Dataset_nogt(opt, detector.pre_process)
    else:
        dataset_val = Heduo_2nd_batch_Dataset(opt, detector.pre_process, opt.val_anno_dir)



    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    refine_3d_model = Refine_3d_easy_Network(5,5)
    load_model(refine_3d_model, opt.refine_model_dir)
    refine_3d_model = refine_3d_model.cuda()

    results = {}
    num_iters = len(dataset_val)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    with torch.no_grad():
        val_loss_total = 0
        val_loss_CenterNetBev = 0
        val_objects_num = 0

        tp, fp, tp2, fp2, fn1, fn2, gtp1, gtp2 = 0, 0, 0, 0, 0, 0, 0, 0

        for ind, (img_id, pre_processed_images) in enumerate(data_loader_val):

            infer_one_img_return = infer_one_img(detector, pre_processed_images, ind, K, D, new_K, bTc, ex4, refine_3d_model)
            if infer_one_img_return == None:
                continue
            else:
                one_img_loss, one_img_loss_CenterNetBev, one_img_objects_num, pred, one_img_objects, one_img_gt, ret = infer_one_img_return
            val_loss_CenterNetBev += one_img_loss_CenterNetBev
            val_objects_num += one_img_objects_num

            val_loss_total += one_img_loss.detach().item()
            print("one_img_loss:", one_img_loss.item())

            np_centernet_bev = one_img_objects.cpu().numpy()
            np_refine_pred = pred.cpu().numpy()
            np_gt = one_img_gt.cpu().numpy()
            if opt.img_nogt_dir == None:
                atp, afp, atp2, afp2, afn1, afn2,agtp1,agtp2 = cal_pr_one_img(np_centernet_bev.copy(),np_refine_pred.copy(),np_gt.copy())
                tp += atp
                fp += afp
                tp2 += atp2
                fp2 +=afp2
                fn1+=afn1
                fn2+=afn2
                gtp1+=agtp1
                gtp2+=agtp2

                im_bev = paint_bev_all(np_centernet_bev.copy(),np_refine_pred.copy(),np_gt.copy())
            else:
                im_bev = paint_bev_nogt(np_refine_pred.copy(), pre_processed_images['image'][0].numpy().copy())
            cv2.imwrite(os.path.join(opt.save_infer_dir,pre_processed_images['img_name'][0]),
                        im_bev)  # todo ziji ttt






            # results[img_id.numpy().astype(np.int32)[0]] = ret['results']
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
                Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                    t, tm=avg_time_stats[t])
            bar.next()

        val_loss_total /= val_objects_num
        val_loss_CenterNetBev /= val_objects_num
        print('val_loss_total',val_loss_total,'val_loss_CenterNetBev',val_loss_CenterNetBev)

        precision1 = tp/(tp+fp)
        precision2 = tp2/(tp2+fp2)
        recall1 = gtp1/(gtp1+fn1)
        recall2 = gtp2/(gtp2+fn2)

        print("precision1",precision1,"precision2",precision2,"recall1",recall1,"recall2",recall2)


    bar.finish()
    # if opt.debug <= 1:
    #     dataset.save_results(results, opt.save_dir)
    # else:
    #     dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().init()
    prefetch_test(opt)