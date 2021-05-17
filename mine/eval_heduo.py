from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

self_coco = coco.COCO("/home/yangfan/project/objectdetection/dataset/heduo/annotations/heduo_5cls_keypoints_val_NoIncomplete_11kps.json")
coco_dets = self_coco.loadRes('{}/results.json'.format("/data1/yangfan/CenterNetExp/multi_pose_0511/dla_3x"))
coco_eval = COCOeval(self_coco, coco_dets, "keypoints")
# coco_eval.params.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
coco_eval.params.kpt_oks_sigmas = np.ones(11).astype(np.float32) #todo ziji
coco_eval.params.maxDets = [40] # keep maxDets pairs of kps per image per category
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
coco_eval = COCOeval(self_coco, coco_dets, "bbox")
# coco_eval.params.maxDets = [1, 10, 50]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()