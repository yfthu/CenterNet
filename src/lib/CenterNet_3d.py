from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.decode import multi_pose_decode

class CenterNet_3d(nn.Module):
    def __init__(self, CenterNet_model, opt):
        super(CenterNet_3d, self).__init__()
        self.CenterNet_model = CenterNet_model
        self.opt = opt
    def forward(self, x):
        CenterNet_outputs, feature_map = self.CenterNet_model(x, return_feature_map=True)

        for k,v in CenterNet_outputs[0].items():
            CenterNet_outputs[0][k] = v.detach()

        with torch.no_grad():
            torch.cuda.synchronize()
            # 来源：multi_pose.py: process函数
            CenterNet_output = CenterNet_outputs[-1]
            CenterNet_output['hm'] = CenterNet_output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                CenterNet_output['hm_hp'] = CenterNet_output['hm_hp'].sigmoid_()

            reg = CenterNet_output['reg'] if self.opt.reg_offset else None
            hm_hp = CenterNet_output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = CenterNet_output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.flip_test:
                raise NotImplementedError() # todo
                CenterNet_output['hm'] = (CenterNet_output['hm'][0:1] + flip_tensor(CenterNet_output['hm'][1:2])) / 2
                CenterNet_output['wh'] = (CenterNet_output['wh'][0:1] + flip_tensor(CenterNet_output['wh'][1:2])) / 2
                CenterNet_output['hps'] = (CenterNet_output['hps'][0:1] +
                                 flip_lr_off(CenterNet_output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

            # print(CenterNet_output['hps'].size()) # [1, 22, 256, 488]
            dets = multi_pose_decode(
                CenterNet_output['hm'], CenterNet_output['wh'], CenterNet_output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)  # reg: 中心点的偏移 wh:bbox的宽和高
            # print(dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[:, :, 5:13])
            # 256 488

            dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])



            # 来源 base_detector.py run 函数124行
            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 3:
                raise NotImplementedError()
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results, img_id, gt=gt)

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}

        ttt = 1