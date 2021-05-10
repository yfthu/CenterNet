import os.path as osp
import mmcv
import re
import os
import cv2
import numpy as np

color_list = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
        (255, 0, 0), (0, 0, 255)]

def add_coco_hp(points, bbox, img_path):
    bbox = [int(i) for i in bbox]
    points = np.array(points, dtype=np.int32).reshape(-1, 2)
    img =cv2.imread(img_path)
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    for j in range(points.shape[0]):
        img = cv2.circle(img, (points[j, 0], points[j, 1]), 3, color_list[j], -1)
        img = cv2.putText(img, str(j),
                          (points[j, 0], points[j, 1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color_list[j], 1, cv2.LINE_AA)
    cv2.imwrite(img_path.split("/")[-1], img)
    print("Saving", img_path.split("/")[-1])
    # for j, e in enumerate(self.edges):
    #     if points[e].min() > 0:
    #         cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
    #                  (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
    #                  lineType=cv2.LINE_AA)


# def add_points(points, img_id='default'):
#     num_classes = len(points)
#     # assert num_classes == len(self.colors)
#     for i in range(num_classes):
#         for j in range(len(points[i])):
#             c = self.colors[i, 0, 0]
#             cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
#                                            points[i][j][1] * self.down_ratio),
#                        5, (255, 255, 255), -1)
#             cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
#                                            points[i][j][1] * self.down_ratio),
#                        3, (int(c[0]), int(c[1]), int(c[2])), -1)

# def vis_show_kps_on_img(foldername, filename, kps):
#     viz = Visdom(env=filename)
#     # H x W x C
#     image1 = io.imread(os.path.join(foldername, filename))
#     # W x H x C
#     viz.image(np.transpose(image1, (2, 0, 1)))

    # viz.image(np.transpose(io.imread('./data/dog.jpg'), (2, 0, 1)), win='dog')

def save_coco_anno(images, annotations, out_file):
    num_of_kps = {1:4, 2:3, 3:2, 4:0, 5:2}
    name_of_cls = {1:"vehicle", 2:"tricycle", 3:"pedestrian", 4:"conebarrel", 5:"bicycle"}
    name_of_kps = {1: ["front_left", "front_right", "rear_right", "rear_left"], 2: ["front", "rear_right", "rear_left"], 3: ["left", "right"], 4: [], 5: ["front", "rear"]}

    categories = []
    for class_i in num_of_kps.keys():
        class_info = dict(
            supercategory=name_of_cls[class_i],
            id=class_i,
            name=name_of_cls[class_i],
            keypoints=name_of_kps[class_i]
        )
        categories.append(class_info)

    if annotations is not None:
        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=categories)
    else:
        coco_format_json = dict(
            images=images,
            categories=categories)


    mmcv.dump(coco_format_json, out_file)


def convert_heduo_to_coco(ann_file, out_file, image_prefix):
    labels = mmcv.list_from_file(ann_file)
    print(len(labels))
    annotations = []
    images = []
    obj_count = 0
    missing_kps_count, incomplete_kps_count = 0, 0
    num_of_kps = {1:4, 2:3, 3:2, 4:0, 5:2}
    max_obj_count = 0

    image_id = -1
    obj_count_perimg = 0
    for line in labels:
        if line.endswith(".jpeg"):
            max_obj_count = max(max_obj_count, obj_count_perimg)
            obj_count_perimg = 0
            # 图片
            image_id += 1
            filename = line
            img_path = osp.join(image_prefix, filename)
            height, width = mmcv.imread(img_path).shape[:2]

            images.append(dict(
                id=image_id,
                file_name=filename,
                height=height,
                width=width))
            # print(image_id)
        else:
            # 标注
            obj_count_perimg += 1
            pattern = re.compile(r'(\d+\.\d*)([eE][-+]?\d+)?')
            reresult = pattern.findall(line)
            floats = [float(x[0]+x[1]) for x in reresult]
            # print(floats)
            # floats = [float(x) for x in floats]
            cat_id = int(line[-1])+1
            if cat_id==5: print("Category \'none\' in filename")
            if cat_id==6: cat_id=5
            kps = floats[4:]
            # print(kps)
            num_kps = num_of_kps[cat_id]
            if kps == []:
                num_kps = 0
                if cat_id != 4:
                    # print("Missing kps in", filename, "for class", cat_id)
                    missing_kps_count += 1
                    kps = [0] * 2*num_of_kps[cat_id]
            else:
                if len(kps) < 2*num_of_kps[cat_id]:
                    # print("Incomplete kps in", filename, "for class", cat_id, "- "+str(int(len(kps)/2))
                    #       + "/" + str(num_of_kps[cat_id]))
                    incomplete_kps_count += 1
                    num_kps = 0
                    kps = [0] * 2 * num_of_kps[cat_id] # erase incomplete keypoints because dont know what they are
                elif len(kps) > 2*num_of_kps[cat_id]:
                    print(kps)
                    add_coco_hp(kps, [floats[0], floats[1], floats[2],floats[3]], os.path.join("/data1/yangfan/heduo/img/", filename))
            # whether pad keypoints
            # elif len(kps)<4:
            #     kps = kps.extend([0]*(4-len(kps)))
            data_anno = dict(
                image_id=image_id,
                category_id=cat_id,
                bbox=[floats[0], floats[1], floats[2],floats[3]],
                area=floats[2]*floats[3],
                num_keypoints=num_kps,
                keypoints=kps,
                id=obj_count
                )
            annotations.append(data_anno)
            obj_count += 1

    max_obj_count = max(max_obj_count, obj_count_perimg)
    save_coco_anno(images, annotations, out_file)
    print("Total images:", image_id+1)
    print("Total objects:", obj_count)
    print("Total sets of keypoints:", obj_count-missing_kps_count)
    print("Full sets of keypoints:", obj_count-missing_kps_count-incomplete_kps_count)
    print("Maximum object per sample:", max_obj_count)

def convert_heduo_to_coco_test(image_folder, out_file):
    test_images = os.listdir(image_folder)
    images = []

    image_id = 7848

    for image_filename in test_images:
        img_path = osp.join(image_folder, image_filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=image_id,
            file_name=image_filename,
            height=height,
            width=width))
        image_id += 1

        save_coco_anno(images, None, out_file)


if __name__ == '__main__':
    convert_heduo_to_coco("/home/lvmengyao/Detection/dataset/heudo/annotations/label_RmRepeatedKps.txt", "/home/lvmengyao/Detection/dataset/heudo/annotations/heduo_5cls_keypoints_NoIncomplete.json", "/data1/yangfan/heduo/img")
    # convert_heduo_to_coco_test("/data1/yangfan/heduo/withoutLabel", "/home/lvmengyao/Detection/dataset/heudo/annotations/heduo_5cls_keypoints_nolabel.json")