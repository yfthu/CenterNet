from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory, get_dataset
from detectors.detector_factory import detector_factory

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func, gt=False):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.gt = gt
    self.task = opt.task
    if self.task == 'holo3d':
      self.cat_ids = dataset.cat_ids
    if self.gt:
      if self.task == 'multi_pose':
        self.num_joints = [4,3,2,0,2]
        if opt.add_kps: self.num_joints = [4,3,2,3,2]
      self.get_ann_ids_func = dataset.coco.getAnnIds
      self.load_anns_func = dataset.coco.loadAnns

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    images, meta = {}, {}
    if self.gt:
      gt = []
      ann_id = self.get_ann_ids_func(imgIds=[img_id])
      anns = self.load_anns_func(ids=ann_id)
      for k in range(len(anns)):
        ann = anns[k]
        if self.task == 'multi_pose': # gt = bbox_kps
          bbox = self._coco_box_to_bbox(ann['bbox'])
          cls_id = int(ann['category_id']) - 1
          all_pts = np.array(ann['keypoints'], np.float32).reshape(sum(self.num_joints), 3)[:, :2]
          # cls_start_idx = [0, 4, 7, -1, 9]
          # pts = np.array(ann['keypoints'], np.float32).reshape(num_kpts_cls, 2)
          # has_kpts = True if np.sum(pts) else False # but have
          # all_pts = np.zeros((sum(self.num_joints), 2), dtype=np.float32)
          # if has_kpts:
          #   all_pts[cls_start_idx[cls_id]:cls_start_idx[cls_id]+num_kpts_cls] = pts.copy()
          dets = np.array(bbox.tolist() + [1.0] + all_pts.reshape(sum(self.num_joints) * 2).tolist()
                          + [cls_id], dtype=np.float32)
        elif self.task == 'holo3d': # gt =
          cls_id = int(self.cat_ids[ann['category_id']])
          if cls_id <= -99:
            continue
          dets = np.array([ct[0], ct[1], 1.0] + \
                               self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
                               [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id], dtype=np.float32)
        else:
          raise NotImplementedError
        gt.append(dets)
    for scale in opt.test_scales:
      if opt.task == 'ddd':
        images[scale], meta[scale] = self.pre_process_func(
          image, scale, img_info['calib'])
      else:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    if self.gt:
      return img_id, {'images': images, 'image': image, 'meta': meta, 'gt': gt}
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  split = opt.split

  Dataset = dataset_factory[opt.dataset]
  # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  # Dataset = get_dataset(opt.dataset, opt.task)
  # opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  dataset = Dataset(opt, split)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]

  opt.save_infer_dir = os.path.join(opt.save_dir, split)
  detector = Detector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process, gt=(opt.debug==2)),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images, img_id=ind)
    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
  bar.finish()
  if opt.debug <= 1:
    dataset.save_results(results, opt.save_dir)
  else:
    dataset.run_eval(results, opt.save_dir)

def test(opt):
  if opt.debug == 2:
    raise NotImplementedError
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  
  split = opt.split
  opt.save_infer_dir = os.path.join(opt.debug_dir, split)
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])

    if opt.task == 'ddd':
      ret = detector.run(img_path, img_info['calib'])
    else:
      ret = detector.run(img_path)
    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  if opt.debug <= 1:
    dataset.save_results(results, opt.save_dir)
  else:
    dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().init()
  if opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)