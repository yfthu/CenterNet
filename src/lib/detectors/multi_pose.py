from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx
    self.save_infer_dir = opt.save_infer_dir
    self.num_classes = opt.num_classes
    self.num_joints = opt.num_joints
    self.all_num_kps = sum(self.num_joints)

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
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
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K) # reg: 中心点的偏移 wh:bbox的宽和高
      # print(dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[:, :, 5:13])
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    # print(dets.size()) # (1,100,28) 4+1+22+1
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    # print(dets.shape) # (1,100,28)
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], self.num_classes, num_joints=self.num_joints)
    if len(dets) > 1:
      raise NotImplementedError
    for j in range(1, self.num_classes+1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5+self.all_num_kps*2)
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes+1):
      results[j] = np.concatenate(
          [detection[j] for detection in detections], axis=0).astype(np.float32)
      if self.opt.nms or len(self.opt.test_scales) > 1:
        soft_nms_39(results[j], Nt=0.5, method=2)
      results[j] = results[j].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:5+self.all_num_kps*2] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results, img_id, gt=None):
    debugger.add_img(image, img_id='multi_pose')
    for cls, bboxes in results.items():
      for bbox in bboxes:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], cls-1, bbox[4], img_id='multi_pose')
          debugger.add_coco_hp(bbox[5:5+self.all_num_kps*2], cls-1, img_id='multi_pose')
          # if cls == 3:
          #   print("draw:", bbox[5:27])
    if gt is not None:
      debugger.add_img(image, img_id='gt')
      for bbox in gt:
        debugger.add_coco_bbox(bbox[0,:4], bbox[0,-1], bbox[0,4], img_id='gt')
        debugger.add_coco_hp(bbox[0,5:5+self.all_num_kps*2], bbox[0,-1], img_id='gt')
        # if bbox[0,-1]+1 == 3:
        #   print("gt:", bbox[0,5:27])

    # debugger.show_all_imgs(pause=self.pause)
    prefix = '{0}_' if isinstance(img_id, str) else '{0:03d}'
    debugger.save_all_imgs(join(self.save_infer_dir, 'VThresh'+str(self.opt.vis_thresh)),
                           prefix=prefix.format(img_id))