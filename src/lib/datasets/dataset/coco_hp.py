from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class COCOHP(data.Dataset):
  # num_classes = 5
  # num_joints = [4,3,2,0,2]
  # default_resolution = [704, 1280]
  # mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3) # todo
  # std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3) # todo
  # flip_idx = [[[0,1], [2,3]],
  #             [[1,2]],
  #             [[0,1]],
  #             [],
  #             []]
  # # "categories": [{"supercategory": "vehicle", "id": 1, "name": "vehicle",
  # #                 "keypoints": ["front_left", "front_right", "rear_right", "rear_left"]},
  # #                {"supercategory": "tricycle", "id": 2, "name": "tricycle",
  # #                 "keypoints": ["front", "rear_right", "rear_left"]},
  # #                {"supercategory": "pedestrian", "id": 3, "name": "pedestrian", "keypoints": ["left", "right"]},
  # #                {"supercategory": "conebarrel", "id": 4, "name": "conebarrel", "keypoints": []},
  # #                {"supercategory": "bicycle", "id": 5, "name": "bicycle", "keypoints": ["front", "rear"]}]}
  # # flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
  # #             [11, 12], [13, 14], [15, 16]]
  # # left and right flip
  # # "categories": [{"keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
  # #                               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
  # #                               "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
  # #                 "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
  # #                              [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}]}

  def __init__(self, opt, split):
    super(COCOHP, self).__init__()
    # self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
    #               [4, 6], [3, 5], [5, 6],
    #               [5, 7], [7, 9], [6, 8], [8, 10],
    #               [6, 12], [5, 11], [11, 12],
    #               [12, 14], [14, 16], [11, 13], [13, 15]]
    
    # self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.data_dir = os.path.join(opt.data_dir, 'heduo')
    self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
    self.default_resolution = opt.default_resolution
    self.num_classes = opt.num_classes
    self.num_joints = opt.num_joints
    self.all_num_kps = sum(self.num_joints)
    self.mean, self.std = opt.mean, opt.std
    self.flip_idx = opt.flip_idx
    if split.startswith('test'):
      self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'heduo_5cls_keypoints_{}.json'.format(split))
          # 'heduo_5cls_keypoints_{}_NoIncomplete.json')
    # elif split == 'val':
    #   self.annot_path = os.path.join(
    #     self.data_dir, 'annotations',
    #     'heduo_5cls_keypoints_{}_NoIncomplete_11kps.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations',
        'heduo_5cls_keypoints_{0}_NoIncomplete_{1}kps.json').format(split, str(self.all_num_kps))
    self.max_objs = 40 # peaks
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32) # same as kitti
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32) # same as kitti
    self.split = split
    self.opt = opt

    print('==> initializing heduo {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()

    self.num_joints.insert(0, 0)
    cls_kps_masks = []
    for i, num_kps in enumerate(self.num_joints[1:]):
      kps_mask = np.zeros(sum(self.num_joints))
      start_idx = sum(self.num_joints[:i+1])
      kps_mask[start_idx: start_idx+num_kps] = 1
      cls_kps_masks.append(kps_mask.reshape(-1,1).astype(np.int32))
      # self.cls_kps_masks.append(kps_mask.reshape(-1,1).repeat(2,1).astype(np.float32))
    self.num_joints.pop(0)

    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = cls_ind # start from 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          kps_vis = np.ones((self.all_num_kps, 1), dtype=np.float32)*cls_kps_masks[category_id-1].reshape(self.all_num_kps, 1)
          keypoints = np.concatenate([
            np.array(dets[5:5+self.all_num_kps*2], dtype=np.float32).reshape(-1, 2),
            kps_vis], axis=1).reshape(self.all_num_kps*3).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))
    print("Results have been saved in ", save_dir)


  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    # coco_eval.params.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    coco_eval.params.kpt_oks_sigmas = np.ones(self.all_num_kps).astype(np.float32)
    coco_eval.params.maxDets = [40] # keep maxDets pairs of kps per image per category
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    # coco_eval.params.maxDets = [1, 10, 50]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# import numpy as np
# from  pycocotools.cocoeval import COCOeval
# import pycocotools.coco as coco
# coco = coco.COCO("/home/lvmengyao/Detection/dataset/heduo/annotations/heduo_5cls_keypoints_val_NoIncomplete_11kps.json")
# coco_dets = coco.loadRes("/home/lvmengyao/Detection/CenterNet/exp/multi_pose/dla_1x_2/debug/val/epoch100/results.json")
# coco_eval = COCOeval(coco, coco_dets, "keypoints")
# coco_eval.params.kpt_oks_sigmas = np.ones(11).astype(np.float32)
# coco_eval.params.maxDets = [40]
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
# coco_eval = COCOeval(coco, coco_dets, "bbox")
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
