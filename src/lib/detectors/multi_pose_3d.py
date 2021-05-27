from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import torch.nn as nn
try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import _nms, _topk, _topk_channel
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger
from models.utils import _gather_feat, _transpose_and_gather_feat
from .base_detector import BaseDetector
from .multi_pose import MultiPoseDetector
from twodtobev import *

def multi_pose_decode(
        heat, wh, kps, feature_map, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2  # 11
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)

    feature_map = _transpose_and_gather_feat(feature_map, inds)
    feature_map = feature_map.contiguous().view(batch, K, -1)

    # print(kps[0, :, 0:8])
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)

        # ziji todo
        # hm_score_factor = (torch.nn.ReLU6()(6*(4 * hm_score - 0.3))) / 6.0 # 0.3 -> 0.9, 0.1->0.1
        # hm_kps = hm_score_factor * hm_kps + (1-hm_score_factor) * kps
        # ziji todo end

        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections,feature_map

class MultiPose3dDetector(MultiPoseDetector):
    def __init__(self, opt):
        super(MultiPose3dDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        # with torch.no_grad():
        torch.cuda.synchronize()
        output, feature_map = self.model(images, return_feature_map = True)
        output = output[-1]

        for k,v in output.items():
            output[k] = v.detach()

        with torch.no_grad():
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

        # print(output['hps'].size()) # [1, 22, 256, 488]
        dets, feature_map = multi_pose_decode(
            output['hm'], output['wh'], output['hps'], feature_map,
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)  # reg: 中心点的偏移 wh:bbox的宽和高
        # print(dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[:, :, 5:13])
        if return_time:
            return output, dets, feature_map, forward_time
        else:
            return output, dets, feature_map

    def run(self, image_or_path_or_tensor, meta=None, img_id=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 4),
                            theme=self.opt.debugger_theme, num_joints=self.opt.num_joints)
        start_time = time.time()
        pre_processed = False
        gt = None
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True
            # gt = image_or_path_or_tensor['bbox_kps_gt'] if self.opt.debug == 2 else None
            # gt = image_or_path_or_tensor['gt_tensor']
            img_name = image_or_path_or_tensor['img_name']

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []

        assert self.scales == [1.0]

        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, feature_map, forward_time = self.process(images, return_time=True)

            assert feature_map.shape[0]==1

            vehicle_mask = dets[0,:,-1]==0
            indices = torch.nonzero(vehicle_mask).reshape((-1))
            vehicle_feature_map = feature_map[0, indices, :] # shape: 27x64 tensor

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 3:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        try:
            vehicle_wheel_points = np.array(results[1])[:, 5:5+8] # shape: 27x8 ndarray
            vehicle_scores = np.array(results[1])[:, 4:5] # shape: 27x1 ndarray
        except IndexError:
            vehicle_feature_map = torch.zeros(size=(0,64))
            vehicle_wheel_points = np.ndarray((0,8))
            vehicle_scores = np.ndarray((0,1))

        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time


        # if self.opt.debug >= 1:
        #     self.show_results(debugger, image, results, img_id, gt=gt)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}, vehicle_feature_map, vehicle_wheel_points, vehicle_scores