            ret, vehicle_feature_map, vehicle_wheel_points, vehicle_scores = detector.run(pre_processed_images, img_id=ind)
            # vehicle_feature_map nx64 Tensor           vehicle_wheel_points nx8 ndarray        vehicle_scores nx1 ndarray
            if vehicle_scores.shape[0] == 0:
                continue
            threshold_mask = (vehicle_scores.reshape(-1) >= OBJECT_THRESHOLD)
            threshold_indices = np.nonzero(threshold_mask)[0]
            vehicle_feature_map = vehicle_feature_map[threshold_indices,:]
            vehicle_wheel_points = vehicle_wheel_points[threshold_indices,:]
            if len(threshold_indices) == 0:
                continue



            vehicle_wheel_points = vehicle_wheel_points.reshape((-1,4,1,2))
            vehicle_wheel_points = [x for x in vehicle_wheel_points]


            undistorted_oneImgObjects = undistort_contours(vehicle_wheel_points, K, D, new_K)
            oneImagePts3d = IPM_contours(undistorted_oneImgObjects, new_K, bTc, ex4,
                                         p=[0, 0, 0, 0.332, 0])  # oneImagePts3d: pandar激光雷达坐标系

            one_img_objects = torch.zeros(size=(len(oneImagePts3d), 5), device='cuda')
            for object_index, oneObject in enumerate(oneImagePts3d):
                keypoints = np.ndarray([4, 2], dtype=float)
                for pts_index in range(4):
                    keypoints[pts_index][0] = oneObject[pts_index][0]  # -25 to 25
                    keypoints[pts_index][1] = oneObject[pts_index][1]  # -50 to 0
                # keypoints x范围 -8到8 右为正方向。 y范围：0到16 上为正方向

                # 估计矩形的初始形状：
                center_x = np.mean(keypoints[:, 0])
                center_y = np.mean(keypoints[:, 1])
                pts_1 = keypoints - np.array([center_x, center_y])
                back_center = (pts_1[2] + pts_1[3]) / 2
                if abs(back_center[0]) < 1e-7:  # ziji todo ttt
                    back_center[0] = 1e-7
                rotation = np.arctan(back_center[1] / back_center[0]) + (np.pi / 2)
                c, s = np.cos(rotation), np.sin(rotation)
                R = np.array([[c, s], [-s, c]], dtype=np.float32)  # 顺时针旋转矩阵
                pts_2 = np.matmul(R, pts_1.T).T
                l = (pts_2[0][1] + pts_2[1][1] - pts_2[2][1] - pts_2[3][1]) / 2
                w = (pts_2[1][0] + pts_2[2][0] - pts_2[0][0] - pts_2[3][0]) / 2

                # 迭代优化：
                x0 = np.array([center_x, center_y, l, w, rotation], dtype=np.float)
                res = minimize(bev_bbox_error, x0, args=keypoints, method='nelder-mead', options={'disp': False})
                # rectangle_final = compute_box_bev(res.x)

                # BGR
                # paint_bev(im_bev, keypoints, (255, 0, 0))
                # paint_bev(im_bev, rectangle_final, (0, 0, 255))
                # objects_bev_pred = np.concatenate((objects_bev_pred, rectangle_final.reshape(1, 4, 2)))

                one_img_objects[object_index] = torch.Tensor(res.x).cuda()

            #one_img_objects Tensor: nx5
            #one_img_ipm = torch.zeros(size=(len(oneImagePts3d), 8), device='cuda')
            one_img_centers = torch.zeros(size=(len(oneImagePts3d), 2), device='cuda')
            for i in range(len(oneImagePts3d)):
                for j in range(4):
                    #one_img_ipm[i, j*2] = oneImagePts3d[i][j][0]
                    #one_img_ipm[i, j*2+1] = oneImagePts3d[i][j][1]
                    one_img_centers[i, 0] += oneImagePts3d[i][j][0]
                    one_img_centers[i, 1] += oneImagePts3d[i][j][1]
            # one_img_ipm: shape nx8 Tensor
            one_img_centers = one_img_centers / 4 # nx2

            #one_img_ipm_featuremap = torch.cat((one_img_ipm, vehicle_feature_map), dim=1) # nx72
            #pred = refine_3d_model(one_img_ipm_featuremap) # nx8
            pred = refine_3d_model(one_img_objects)  # nx5

            one_img_gt = pre_processed_images['gt_tensor'][0].cuda()
            one_img_loss = torch.zeros(1,requires_grad=True).cuda()
            for one_object_pred_index in range(pred.shape[0]):
                min_distance = 99999999
                min_index = None
                for one_object_gt_index in range(one_img_gt.shape[0]):
                    cur_dis_2 = (one_img_centers[one_object_pred_index, 0] - one_img_gt[one_object_gt_index, 0]) ** 2 \
                    + (one_img_centers[one_object_pred_index, 1] - one_img_gt[one_object_gt_index, 1]) ** 2
                    if cur_dis_2 < min_distance:
                        min_index = one_object_gt_index
                        min_distance = cur_dis_2
                # cal loss
                if min_distance > 25: # ziji todo
                    continue
                if min_index != None:
                    diff = nn.MSELoss(reduction='none')(pred[one_object_pred_index],one_img_gt[min_index])
                    diff[4] *= 10
                    one_object_loss = diff.sum()
                    one_img_loss += one_object_loss

                    diff2 = nn.MSELoss(reduction='none')(one_img_objects[one_object_pred_index], one_img_gt[min_index])
                    diff2[4] *= 10
                    one_object_loss2 = diff2.sum()
                    train_loss_CenterNetBev += one_object_loss2.item()
                    print('one_object_loss', one_object_loss.item(), 'one_object_loss2', one_object_loss2.item())
                    objects_num += 1