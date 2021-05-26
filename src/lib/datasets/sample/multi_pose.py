from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

class MultiPoseDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)


    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints # a list of kps for 0-4 classes
    all_num_kps = sum(num_joints)
    cls_start_idx = [sum(num_joints[:i]) for i in range(len(num_joints))] # [0, 4, 7, 9, 9] # [0, 4, 7, 9, 12]
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    # max_objs: how many targets might be present in one sample
    # all_num_kps: including heads of all classes, 4 for vehicle, 2 for pedestrain, ...

    # object level: center, local offset, object size
    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32) # bbox center
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # bbox width and height
    ind = np.zeros((self.max_objs), dtype=np.int64) # index of center if image flatten
    reg = np.zeros((self.max_objs, 2), dtype=np.float32) # float center - int center = offset
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

    # keypoint level: kps locations, kps heatmap, kps offset
    kps = np.zeros((self.max_objs, all_num_kps * 2), dtype=np.float32) # joint location from bbox center
    kps_mask = np.zeros((self.max_objs, all_num_kps * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * all_num_kps, 2), dtype=np.float32) # joint offset
    hp_ind = np.zeros((self.max_objs * all_num_kps), dtype=np.int64) # joint offset from center
    hp_mask = np.zeros((self.max_objs * all_num_kps), dtype=np.int64)
    hm_hp = np.zeros((all_num_kps, output_res, output_res), dtype=np.float32) # joint heatmap

    dense_kps = np.zeros((all_num_kps, 2, output_res, output_res), dtype=np.float32) #
    dense_kps_mask = np.zeros((all_num_kps, output_res, output_res), dtype=np.float32) # joint heatmap

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    # for visualization
    bbox_kps_cls = np.zeros((self.max_objs, 5+all_num_kps*2+1), dtype=np.float32)

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      # if cls_id != 3:
      #   pass
      num_kpts_cls = num_joints[cls_id] # should have
      try:
        pts = np.array(ann['keypoints'], np.float32).reshape(all_num_kps, 3)
      except ValueError as ve:
        print(bbox, ann['keypoints'], num_kpts_cls)
      # all_pts = np.zeros((all_num_kps, 2), dtype=np.float32)
      has_kpts = pts[:, 2].sum() > 0 # but have
      pts = pts[:, :2]
      # print(cls_id, num_kpts_cls, has_kpts)
      # if has_kpts:
      #   try:
      #     all_pts[cls_start_idx[cls_id]:cls_start_idx[cls_id]+num_kpts_cls] = pts.copy()
      #   except ValueError as ve:
      #     print(pts, all_pts, cls_start_idx[cls_id], cls_start_idx[cls_id]+num_kpts_cls)
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        if has_kpts:
          pts[:, 0] = width - pts[:, 0] - 1
          for e in self.flip_idx[cls_id]:
            pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_res + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        # num_kpts = pts[:, 2].sum()
        if not has_kpts:
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999 # ?
          reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
        # two considerations: no kps like class bucket, and missing kps
        if has_kpts:
          for j in range(cls_start_idx[cls_id], cls_start_idx[cls_id] + num_kpts_cls):
            pts[j] = affine_transform(pts[j], trans_output_rot)
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j] - ct_int
              kps_mask[k, j * 2: j * 2 + 2] = 1
              # here the mask treats keypoints not belonging to the class, and kps not existing equally
              pt_int = pts[j].astype(np.int32)
              hp_offset[k * all_num_kps + j] = pts[j] - pt_int
              hp_ind[k * all_num_kps + j] = pt_int[1] * output_res + pt_int[0]
              hp_mask[k * all_num_kps + j] = 1
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)

        draw_gaussian(hm[cls_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1] +
                       pts.reshape(all_num_kps * 2).tolist() + [cls_id])
        bbox_kps_cls[k] = np.array([ct[0] - w / 2, ct[1] - h / 2,
                                   ct[0] + w / 2, ct[1] + h / 2, 1] +
                                   pts.reshape(all_num_kps * 2).tolist()
                                   + [cls_id], dtype=np.float32)

    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'hps': kps, 'hps_mask': kps_mask, 'bbox_kps_cls': bbox_kps_cls}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 5 + all_num_kps * 2 + 1), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    # print(inp.shape) # (3,1280,1280)
    return ret
