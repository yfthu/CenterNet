# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import json
from glob import glob, iglob
import math
import random
import shutil

random.seed(311)

import _init_paths
# from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image3,alpha2rot_y
# from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
from utils.projections import LoadCameraParamsRaw, trucate_angle, Gen3DBox, project

DATA_PATH = './data/heduo-2/images/'
LABEL_PATH = './data/heduo-2/labels/'
ANNO_PATH = './data/heduo-2/annotations/'
SPLIT_PATH = './data/heduo-2/splits/'
DEBUG = False
# prioritize 3d projection result, if nan exists, ignore.
# if negative points are projected, keep.
# this is to cooperate with the projection during later training
INCONSISTENT_DUMP = True # if any(diff(3dbox_pj, 3dbox_gt))>THRESH, dump the box instead of using the GT box
# even under this condition the GT 3d box looks right, we dont use those objects because it may bring troubles later
KEEP_DIFF_THRESH = 5.0 # for

SPLITS = ['all', 'train', 'val', 'trainval', 'test']
NO_ALL_LABELS = 9396

cats = ['Car', 'Truck', 'Engineering_truck', 'Bus', 'Tricycle', 'Rider', 'Bicycle',
        'Biker', 'Motorcycle', 'Pedestrian', 'Cone', 'Ignore', 'Other']
det_cats = cats #['Car', 'Pedestrian', 'Cyclist']

cat_ids = {cat[:3].lower(): i + 1 for i, cat in enumerate(cats)} # starts from 1

cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

def divide_dataset(ratio=[0.5, 0.2, 0.3]):
    alist = list(range(NO_ALL_LABELS))
    split_idx_dict = {'all': alist}
    assert sum(ratio) == 1
    num_images = [int(i*NO_ALL_LABELS) for i in ratio]
    num_images[-1] = NO_ALL_LABELS - sum(num_images[:-1])
    print("Dataset split: ", num_images)
    trainlist = random.choices(alist, k=num_images[0])
    split_idx_dict['train'] = trainlist
    alist = list(set(alist) - set(trainlist))
    vallist = random.choices(alist, k=num_images[1])
    split_idx_dict['val'] = vallist
    split_idx_dict['trainval'] = trainlist + vallist
    alist = list(set(alist) - set(vallist))
    split_idx_dict['test'] = alist
    return split_idx_dict

def read_calib_kitti(calib_path, line_num):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == line_num:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

def read_calib_yaml(calib_path):
    with open(calib_path, 'r') as yamlfile:
        pass

def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

color_list = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
              (255, 0, 255),
              (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
              (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
              (255, 0, 0), (0, 0, 255)]

def Draw3DBox(image, pts):
    clr = (0, 255, 0)
    rlc = (255, 0, 255)
    cv2.line(image, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), clr)
    cv2.line(image, (pts[1, 0], pts[1, 1]), (pts[2, 0], pts[2, 1]), clr)
    cv2.line(image, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), clr)
    cv2.line(image, (pts[3, 0], pts[3, 1]), (pts[0, 0], pts[0, 1]), clr)

    cv2.line(image, (pts[4, 0], pts[4, 1]), (pts[5, 0], pts[5, 1]), rlc)
    cv2.line(image, (pts[5, 0], pts[5, 1]), (pts[6, 0], pts[6, 1]), rlc)
    cv2.line(image, (pts[6, 0], pts[6, 1]), (pts[7, 0], pts[7, 1]), rlc)
    cv2.line(image, (pts[7, 0], pts[7, 1]), (pts[4, 0], pts[4, 1]), rlc)

    cv2.line(image, (pts[0, 0], pts[0, 1]), (pts[4, 0], pts[4, 1]), clr)
    cv2.line(image, (pts[1, 0], pts[1, 1]), (pts[5, 0], pts[5, 1]), clr)
    cv2.line(image, (pts[2, 0], pts[2, 1]), (pts[6, 0], pts[6, 1]), clr)
    cv2.line(image, (pts[3, 0], pts[3, 1]), (pts[7, 0], pts[7, 1]), clr)
    if pts.shape[0] == 9:
        cv2.circle(image, (int(pts[8, 0]), int(pts[8, 1])), 5, (255, 0, 0), -1)

def add_coco_hp(pointss, bboxes, img, img_path, draw_3dbox):
    for count, (points, bbox) in enumerate(zip(pointss, bboxes)):
        if np.isnan(np.sum(bbox)):
            print("NaN in bbox!")
            continue
        bbox = [int(i) for i in bbox]
        points = points.astype(np.int32)[:, :2]
        if draw_3dbox:
            Draw3DBox(img, points)
        else:
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        for j in range(points.shape[0]):
            img = cv2.circle(img, (points[j, 0], points[j, 1]), 3, color_list[j], -1)
            if not draw_3dbox:
                img = cv2.putText(img, str(j),
                                  (points[j, 0], points[j, 1]), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, color_list[j], 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join("/data1/lvmengyao/tmp/", img_path.split("/")[-1]), img)
    print("Saving", img_path.split("/")[-1])


def convert_holo_to_coco(view='', divide='523'):
    # view_convert = {"前视": "center", "左视": "left", "右视": "right"}
    # image_paths = []
    # subfolder_names = [view_convert[view], "img"]
    # for subfolder_name in subfolder_names:
    #     image_paths.extend(glob(os.path.join(DATA_PATH, "*"+view+"_0[0-9]/"+subfolder_name+"/*.jpeg")))
    # label_paths = glob(os.path.join(LABEL_PATH, "*"+view+"_0[0-9]_result/*.json"))
    label_paths = glob(os.path.join(LABEL_PATH, "*.json"))
    print("Found {:0d} label files.".format(len(label_paths)))
    # tmp = [i.split('/')[-2] for i in label_paths]
    # print("Searched following folders: ", set(tmp))
    # calib_path = "/data1/yangfan/KITTI/object/testing/calib/004384.txt" # todo
    camera = LoadCameraParamsRaw("./src/tools/center.json")
    splits_path = os.path.join(SPLIT_PATH, divide)
    if not os.path.exists(splits_path): os.mkdir(splits_path)
    divide_list = [0.1*int(i) for i in list(divide)]
    print("Spliting the dataset into ", divide_list)
    split_idx_dict = divide_dataset(ratio=divide_list)
    for split, split_idx in split_idx_dict.items():
        if split != 'all':
            split_path = os.path.join(splits_path, split)
            if os.path.exists(split_path): shutil.rmtree(split_path)
            os.mkdir(split_path)
        else:
            continue
        label_path_split = [label_paths[i] for i in split_idx]
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_id = -1
        # missing_3dbox = 0
        # normal_max_diff = 0
        for count, label_path in enumerate(label_path_split):
            # image
            image_id += 1
            tmp = json.loads(open(label_path).read())
            if len(tmp) > 1: raise NotImplementedError
            tmp = tmp[0]
            img_path = tmp.get('img', None)
            if img_path is not None:
                img_path = img_path.split("/")[-1]
                if split != 'all': shutil.copyfile(os.path.join(DATA_PATH, img_path), os.path.join(split_path, img_path))
            else:
                print("No image path in ", label_path)
                raise NotImplementedError

            # calib0 = read_calib_kitti(calib_path, 0)
            # calib = read_calib_kitti(calib_path, 2) # shape (3,4)
            # calib = np.array([[ -946.78235613, -932.56498948, 139.56820188, -796.00704443],
            #                           [   29.53376955, -738.02077588, -842.12387376, -1050.25294344],
            #                           [    0.00496699, -0.98443095,    0.17570157,   -0.73524135]], dtype=np.float32)
            image_info = {'file_name': '{}'.format(img_path),
                          'id': int(image_id),
                          'calib': None
                          }
            ret['images'].append(image_info)

            # annotations
            if DEBUG:
                pointss, bboxes = [], []
            object_labels = tmp.get(u'\u6709\u70b9\u4e91\u6846', None)
            if DEBUG: img_save = False
            if object_labels is not None:
                try:
                    image = cv2.imread(os.path.join(DATA_PATH, img_path))
                except:
                    print(os.path.join(DATA_PATH, img_path))
                if image is None:
                    print(os.path.join(DATA_PATH, img_path))
                for object in object_labels:
                    cat_ini = object['attributes'][u'\u5c5e\u6027'][0][:3]
                    cat_id = cat_ids[cat_ini]
                    dim = [float(object["width"]), float(object["height"]), float(object["depth"])]
                    l, w, h = dim # should be w, h, l
                    location = [float(object["center"]["x"]), float(object["center"]["y"]), float(object["center"]["z"])]
                    # off_set=(calib[0,3]-calib0[0,3])/calib[0,0]
                    # location[0] += off_set###################################################confuse
                    rotation_y = float(object["rotation"]["z"])
                    alpha = trucate_angle(rotation_y - np.pi/2)
                    if object["rotation"]["x"] != 0 or object["rotation"]["y"] != 0:
                        raise NotImplementedError


                    box_3d = Gen3DBox(*location, w, h, l, alpha) # (9,2)
                    box_3dto2d_pj, num_keypoints = project(box_3d, camera) # (9, 3)

                    # check points status
                    if np.isnan(np.sum(box_3dto2d_pj)): continue # ignore
                    box_3d_dict = object['attributes'].get('立体框')
                    if box_3d_dict is None: # load GT 3D box to check consistency
                        if DEBUG:
                            img_save = True
                            print("No 3D GT box but the projected has no NaN for {}, object {}, class {}".format(label_path, object["id"], cat_ini))
                        else:
                            continue
                        # checked all 3 images under this condition, the last one 20201109_1321191_center_3D_00011810.jpeg is strange
                    else:
                        box_3d_dict = list(box_3d_dict[0].values())
                        box_3dto2d_gt = []
                        for point in box_3d_dict:
                            box_3dto2d_gt.extend([float(point['x']), float(point['y'])])
                        box_3dto2d_gt = np.array(box_3dto2d_gt, dtype=np.float32).reshape(-1, 2) # (8,2)
                        box_3dto2d_gt = box_3dto2d_gt[[6, 7, 5, 4, 2, 3, 1, 0]] # ('abcdefgh->ghfecdba'), consistent with projected
                        box_3dto2d_diff = np.absolute(box_3dto2d_pj[:8, :2] - box_3dto2d_gt)
                        # if np.min(box_3dto2d_gt) <= 0 or np.max(box_3dto2d_gt[:, 0])>=1920 or np.max(box_3dto2d_gt[:, 1])>=1020:
                        #     if np.max(box_3dto2d_diff) > 1.0:
                        #     else:
                        #         print("OOB but right project for {}, object {}, class {}".format(label_path, object["id"], cat_ini))
                        #         # if DEBUG: img_save = True
                        # else:
                        if np.max(box_3dto2d_diff) > KEEP_DIFF_THRESH:
                            if DEBUG: img_save = True
                            else: continue
                            # normal_max_diff = max(normal_max_diff, np.max(box_3dto2d_diff)) # 236.494873046875

                    # l, t, r, b
                    box_2d = [box_3dto2d_pj[:, 0].min(), box_3dto2d_pj[:, 1].min(), box_3dto2d_pj[:, 0].max(), box_3dto2d_pj[:, 1].max()]
                    box_2d = [float(i) for i in box_2d]
                    if DEBUG:
                        box_2d_gt = [box_3dto2d_gt[:, 0].min(), box_3dto2d_gt[:, 1].min(), box_3dto2d_gt[:, 0].max(), box_3dto2d_gt[:, 1].max()]
                        box_2d_gt = [float(i) for i in box_2d_gt]

                    # if box_3d_dict is None:
                    #     print("No 3D box in {}, object {}, class {}".format(label_path, object["id"], cat_ini))
                    #     missing_3dbox += 1
                    #     # get 2D box from projected 3D bbox
                    #     box_3d = Gen3DBox(*location, w, h, l, alpha) # (9,2)
                    #     box_3dto2d, num_keypoints = project(box_3d, camera) # (9, 3)
                    #     if np.isnan(np.sum(box_3dto2d)):
                    #         print("NaN points for image {}, object {}, class {}".format(img_path, object["id"], cat_ini))
                    #         # ignore if at least one part of the object is behind the camera
                    #         continue
                    #     else:
                    #         if DEBUG: img_save = True
                    # else:
                    #     # get 2D box from 3D bbox given by anno files
                    #     box_3d_dict = list(box_3d_dict[0].values())
                    #     box_3dto2d = []
                    #     for point in box_3d_dict:
                    #         box_3dto2d.extend([float(point['x']), float(point['y'])])
                    #     box_3dto2d_gt = np.array(box_3dto2d, dtype=np.float32).reshape(-1, 2) # (8,2)
                    #     box_3dto2d = box_3dto2d_gt[[6, 7, 5, 4, 2, 3, 1, 0]] # ('abcdefgh->ghfecdba')
                    #     # if projected points have nan, the GT points should have <0, or > 1920x1020
                    #     if DEBUG:
                    #         box_3d = Gen3DBox(*location, w, h, l, alpha) # (9,2)
                    #         box_3dto2d_pj, num_keypoints = project(box_3d, camera) # (9, 3)
                    #         if np.isnan(np.sum(box_3dto2d_pj)):
                    #             print("NaN points for image {}, object {}, class {}".format(img_path, object["id"], cat_ini))
                    #             if np.min(box_3dto2d_gt) < 0 or np.max(box_3dto2d_gt[:, 0]<1920) or np.max(box_3dto2d_gt[:, 1]<1020):
                    #                 pass
                    #             else:
                    #                 print("inconsistent GT and projection!")
                    #                 img_save = True
                    #     if np.isnan(np.sum(box_3dto2d_pj)): continue # ignore even if the GT anno file has provided negative positions
                    # # l, t, r, b
                    # box_2d = [box_3dto2d[:, 0].min(), box_3dto2d[:, 1].min(), box_3dto2d[:, 0].max(), box_3dto2d[:, 1].max()]
                    # box_2d = [float(i) for i in box_2d]

                    if DEBUG:
                        # if bbox[1] < 0: # or bbox[3] > 1020:
                        #     continue
                        # if (box_3d.min() < 0):
                            # print("Below zero 3D kps!")
                            # img_save = True
                        pointss.append(box_3dto2d_pj)
                        bboxes.append(_bbox_to_coco_bbox(box_2d))
                        pointss.append(box_3dto2d_gt)
                        bboxes.append(_bbox_to_coco_bbox(box_2d_gt))

                    truncated = 0 # non-truncated
                    occluded = 0 # fully visible
                    ann = {'image_id': image_id, # done
                           'id': int(len(ret['annotations']) + 1), # object["id"], # done, not int, str!
                           'category_id': cat_id, # done
                           'dim': dim,  # dim
                           'bbox': _bbox_to_coco_bbox(box_2d), # done
                           'depth': location[2],
                           'alpha': alpha,
                           'truncated': truncated,
                           'occluded': occluded,
                           'location':location,
                           'rotation_y': rotation_y  # done
                           }
                    ret['annotations'].append(ann)
                if DEBUG and img_save:
                    add_coco_hp(pointss, bboxes, image, img_path, draw_3dbox=True)

        out_path = '{}/heduo-2-{}-{}.json'.format(ANNO_PATH, divide, split)
        json.dump(ret, open(out_path, 'w'))
        print("Num of images for split {}: {}".format(split, len(ret['images'])))
        print("Num of objects for split {}: {}".format(split, len(ret['annotations'])))
        # print(normal_max_diff) # 6.103515625e-05

def calculate_cats(view=''):
    label_list = glob(os.path.join(LABEL_PATH, "*" + view + "*_result/*.json")) # not includes sequences
    print("Total num of images: ", len(label_list))
    object_counts = 0
    no_anno_image_counts = 0
    max_anno_per_image = 0
    cat_counts = {}
    for image_label in label_list:
        tmp = json.loads(open(image_label).read())[0]
        object_labels = tmp.get('有点云框', None)
        if object_labels is not None:
            num_of_objects = len(object_labels)
            object_counts += num_of_objects
            max_anno_per_image = max(max_anno_per_image, num_of_objects)
            for object in object_labels:
                cat = object['attributes']['属性'][0]
                cat_counts[cat] = 1 + cat_counts.get(cat, 0)
        else:
            no_anno_image_counts += 1
    print("no anno image counts: ", no_anno_image_counts)
    print("object counts: ", object_counts)
    print("cats: ", cat_counts)
    print("max anno per image: ", max_anno_per_image)


if __name__ == '__main__':
    # calculate_cats(view='前视')
    divides = ['523', '325', '217']
    for divide in divides:
        convert_holo_to_coco(divide=divide)