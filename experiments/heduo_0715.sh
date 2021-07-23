# 老以前train的dla
python main.py multi_pose --exp_id dla_1x_3 --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 3,4 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_1x_3 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --display_port 8098


# lmy 运行的train脚本：
python main.py multi_pose --exp_id res_18_14kps --arch res_18 --head_conv 64  --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 4,5,6 --num_workers 8 --master_batch 2 --debug 5 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --add_kps

# 我 train的脚本
python main.py multi_pose --exp_id res_18_14kps --arch res_18 --head_conv 64  --dataset coco_hp --batch_size 7 --lr 3.12e-5 --gpus 4,7 --num_workers 8 --master_batch 3 --debug 5 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --add_kps

# 3d测试
CUDA_VISIBLE_DEVICES=3 python test_3d_val.py multi_pose_3d --exp_id res_18_new0716_lmy --arch res_18 --head_conv 64 --add_kps  --dataset coco_hp --keep_res --load_model "/data1/lvmengyao/CenterNetExp/multi_pose/res_18_14kps_ac2/model_236.pth" --refine_model_dir "/data1/yangfan/CenterNetExp/multi_pose_3d/dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10/model_Refine3d_38_4.651825075710478.pth" --split val --vis_thresh 0.3 --debug 2 --K 20 --refine_loss l1 --match_max_center_dis 10.0 --object_threshold 0.3
CUDA_VISIBLE_DEVICES=2 python test_3d_val.py multi_pose_3d --exp_id res_18_new0716 --arch res_18 --head_conv 64 --add_kps  --dataset coco_hp --keep_res --load_model "/data1/yangfan/CenterNetExp/multi_pose/res_18_14kps/model_last_225.pth" --refine_model_dir "/data1/yangfan/CenterNetExp/multi_pose_3d/dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10/model_Refine3d_38_4.651825075710478.pth" --split val --vis_thresh 0.3 --debug 2 --K 20 --refine_loss l1 --match_max_center_dis 10.0 --object_threshold 0.3

# 2d bbox测试脚本
CUDA_VISIBLE_DEVICES=2 python test.py multi_pose --exp_id res_18_new0716_lmy --arch res_18 --head_conv 64 --add_kps  --dataset coco_hp --keep_res --load_model "/data1/lvmengyao/CenterNetExp/multi_pose/res_18_14kps_ac2/model_236.pth" --split val --vis_thresh 0.3 --debug 2
CUDA_VISIBLE_DEVICES=1 python test.py multi_pose --exp_id resnet_18_14kps_origin_0625_lmy --arch res_18 --head_conv 64 --add_kps  --dataset coco_hp --keep_res --load_model "/data1/yangfan/CenterNetExp/multi_pose/resnet_18_14kps_origin_0625_lmy.pth" --split val --vis_thresh 0.3 --debug 2
CUDA_VISIBLE_DEVICES=2 python test.py multi_pose --exp_id res_18_new0716 --arch res_18 --head_conv 64 --add_kps  --dataset coco_hp --keep_res --load_model "/data1/yangfan/CenterNetExp/multi_pose/res_18_14kps/model_last_225.pth" --split val --vis_thresh 0.3 --debug 2