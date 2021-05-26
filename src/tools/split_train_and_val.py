import os,shutil
import mmcv
import re

root_path = "/data1/yangfan/heduo/"
data_infos = mmcv.load("/home/lvmengyao/Detection/dataset/heduo/annotations/heduo_5cls_keypoints_NoIncomplete_14kps.json")

images_name_val = os.listdir(root_path+'val')

trainimages = []
trainannos = []
valimages = []
valannos = []

val_id = []


for oneDist in data_infos['images']:
    if oneDist['file_name'] in images_name_val:
        valimages.append(oneDist)
        val_id.append(oneDist['id'])
    else:
        trainimages.append(oneDist)

for oneDist in data_infos['annotations']:
    oneDist['iscrowd'] = 0
    if oneDist['image_id'] in val_id:
        valannos.append(oneDist)
    else:
        trainannos.append(oneDist)

train_dict = dict(
        images=trainimages,
        annotations=trainannos,
        categories=data_infos['categories'])

val_dict = dict(
        images=valimages,
        annotations=valannos,
        categories=data_infos['categories'])

mmcv.dump(train_dict, "/home/lvmengyao/Detection/dataset/heduo/annotations/heduo_5cls_keypoints_train_NoIncomplete_14kps.json")
mmcv.dump(val_dict, "/home/lvmengyao/Detection/dataset/heduo/annotations/heduo_5cls_keypoints_val_NoIncomplete_14kps.json")
