from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import ddd_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from .ddd import DddLoss

class HoloTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(DddTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss',
                       'wh_loss', 'off_loss']
        loss = DddLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        # calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        # if opt.dataset == 'gta':
        #   dets[:, 12:15] /= 3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), None, opt)
        dets_gt = ddd_post_process(
            batch['meta']['gt_det'].detach().numpy().copy(),
            batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), None, opt)
        #for i in range(input.size(0)):
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==4),
                                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'hm_pred')
            debugger.add_blend_img(img, gt, 'hm_gt')
            # decode
            debugger.add_ct_detection(
                img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh,
                img_id='det_pred')
            debugger.add_ct_detection(
                img, batch['meta']['gt_det'][i].cpu().numpy().copy(),
                show_box=opt.reg_bbox, img_id='det_gt')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_pred')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_gt[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_gt')
            # debugger.add_bird_view(
            #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
            # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
            debugger.add_bird_views(
                dets_pred[i], dets_gt[i],
                center_thresh=opt.center_thresh, img_id='bird_pred_gt')

            # debugger.add_blend_img(img, pred, 'out', white=True)
            debugger.compose_vis_add(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
            # debugger.add_img(img, img_id='out')
            if opt.debug == 5:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        opt = self.opt
        wh = output['wh'] if opt.reg_bbox else None
        reg = output['reg'] if opt.reg_offset else None
        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        dets_pred = ddd_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        img_id = batch['meta']['img_id'].detach().numpy()[0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
            results[img_id][j] = results[img_id][j][keep_inds]