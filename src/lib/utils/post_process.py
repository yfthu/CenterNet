from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot
from .projections import LoadCameraParamsRaw

def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  if calibs is not None:
    dets = ddd_post_process_3d(dets, calibs)
  else:
    pass # todo for holo
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w, num_classes=5, num_joints=[4,3,2,0,2]):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord, 4+1+17*2
  all_num_kps = sum(num_joints)
  num_joints_a = num_joints
  num_joints_a.insert(0, 0)
  cls_kps_masks = []
  for i, num_kps in enumerate(num_joints_a[1:]):
    kps_mask = np.zeros(sum(num_joints_a))
    start_idx = sum(num_joints_a[:i+1])
    kps_mask[start_idx: start_idx+num_kps] = 1
    cls_kps_masks.append(kps_mask.reshape(-1,1).astype(np.int32))
  num_joints_a.pop(0)
  # cls_kps_masks = [np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
  #                  np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32),
  #                  np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=np.int32),
  #                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
  #                  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32)]
  ret = []
  # print("dets shape:", dets.shape) # (1, 100, 28)
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h)).reshape(-1, 4) # 200,2 -> 100,4
    pts = transform_preds(dets[i, :, 5:5+all_num_kps*2].reshape(-1, 2), c[i], s[i], (w, h)).reshape(-1, all_num_kps*2) # 1100,2 -> 100,22
    conf = dets[i, :, 4:5] # 100,1
    # cls_ids = dets[i, :, -1]
    ret_classes = {}
    for j in range(num_classes):
      mask_j = cls_kps_masks[j].reshape(all_num_kps,1).repeat(2,1).reshape(1,-1)
      idx_j = dets[i][:, -1]==j
      top_preds = np.concatenate(
        [bbox[idx_j], conf[idx_j],
         pts[idx_j]*mask_j], axis=1).astype(np.float32).tolist()
      ret_classes[j+1] = top_preds
      # if j==2:
      #   print("dets:", dets[i, :, 5:27][idx_j]*mask_j[0])
      #   print("transformed:", pts[idx_j]*mask_j[0])
    ret.append(ret_classes)
      #dets: 0:4 bbox, 4:5 confidence, 5:27 pts
  return ret
