from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # todo

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
from twodtobev import undistort_contours, IPM_contours, cam_intrinsic, cam_extrinsic, compute_box_bev
from refine_3d_easy_network import Refine_3d_easy_Network
from models.model import save_model
from visdom import Visdom
from torch.optim.lr_scheduler import CosineAnnealingLR
from util_3d import *

def prefetch_test(opt):
    viz = Visdom(env=opt.exp_id, port=8098)
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

    refine_3d_model = Refine_3d_easy_Network(5,5).cuda()

    #optimizer = torch.optim.Adam(chain(detector.model.parameters(), refine_3d_model.parameters()), lr=1e-4)
    optimizer = torch.optim.Adam(refine_3d_model.parameters(), lr=1e-4)
    scheduler_1 = CosineAnnealingLR(optimizer, T_max=40)



    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    val_loss_min = 1e30
    for epoch in range(40): # todo ziji
        # train
        train_loss_total = 0
        train_loss_CenterNetBev = 0
        objects_num = 0
        for ind, (img_id, pre_processed_images) in enumerate(data_loader):
            optimizer.zero_grad()

            infer_one_img_return = infer_one_img(detector, pre_processed_images, ind, K, D, new_K, bTc, ex4,
                                                 refine_3d_model)
            if infer_one_img_return == None:
                continue
            else:
                one_img_loss, one_img_loss_CenterNetBev, one_img_objects_num, pred, one_img_objects, one_img_gt, ret = infer_one_img_return
            train_loss_CenterNetBev += one_img_loss_CenterNetBev
            objects_num += one_img_objects_num



            one_img_loss.backward()
            optimizer.step()

            train_loss_total += one_img_loss.detach().item()
            print("one_img_loss:" ,one_img_loss.item())

            # results[img_id.numpy().astype(np.int32)[0]] = ret['results']
            Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
                Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                    t, tm=avg_time_stats[t])
            bar.next()

        scheduler_1.step()
        train_loss_total /= objects_num
        train_loss_CenterNetBev /= objects_num
        viz.line(
            X=[epoch],
            Y=[[train_loss_total,train_loss_CenterNetBev]],
            win='train_loss',
            opts=dict(title='train_loss', legend=['train_loss_total','train_loss_CenterNetBev']),
            update='append')

        if epoch % 2 == 0:
            # val
            with torch.no_grad():
                val_loss_total = 0
                val_loss_CenterNetBev = 0
                val_objects_num = 0
                for ind, (img_id, pre_processed_images) in enumerate(data_loader_val):
                    infer_one_img_return = infer_one_img(detector, pre_processed_images, ind, K, D, new_K, bTc, ex4,
                                                         refine_3d_model)
                    if infer_one_img_return == None:
                        continue
                    else:
                        one_img_loss, one_img_loss_CenterNetBev, one_img_objects_num, pred, one_img_objects, one_img_gt, ret = infer_one_img_return
                    val_loss_CenterNetBev += one_img_loss_CenterNetBev
                    val_objects_num += one_img_objects_num

                    val_loss_total += one_img_loss.detach().item()
                    print("one_img_loss:", one_img_loss.item())

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
                viz.line(
                    X=[epoch],
                    Y=[[val_loss_total, val_loss_CenterNetBev]],
                    win='val_loss',
                    opts=dict(title='val_loss', legend=['val_loss_total', 'val_loss_CenterNetBev']),
                    update='append')

                if val_loss_total < val_loss_min:
                    val_loss_min = val_loss_total
                # save_model(os.path.join(opt.save_dir, 'model_CenterNet_{}_{}.pth'.format(epoch, val_loss_total)),
                # epoch, detector.model, optimizer)
                save_model(os.path.join(opt.save_dir, 'model_Refine3d_{}_{}.pth'.format(epoch, val_loss_total)),
                           epoch, refine_3d_model, optimizer)

    bar.finish()
    # if opt.debug <= 1:
    #     dataset.save_results(results, opt.save_dir)
    # else:
    #     dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().init()
    prefetch_test(opt)