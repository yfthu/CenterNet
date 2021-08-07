# 2d
# 训练
CUDA_VISIBLE_DEVICES=2,3,4,5,6 python main.py multi_pose --exp_id hg_1x_heduo --dataset coco_hp --arch hourglass --batch_size 18 --master_batch 2 --lr 1.875e-4 --load_model ../models/multi_pose_hg_1x.pth --gpus 0,1,2,3,4 --num_epochs 50 --lr_step 40 --K 40 --input_h 480 --input_w 640 --debug 5 --display_env hg_1x_heduo --display_port 8098
# 验证
CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id hg_1x_heduo --dataset coco_hp --keep_res --load_model ../exp/multi_pose/hg_1x_heduo/model_last.pth --split val --vis_thresh 0.3 --debug 2 --input_h 480 --input_w 640
# 推断
CUDA_VISIBLEVICES=3 python demo.py multi_pose --exp_id data_3rd_batch_infer_2d --demo /data1/yangfan/heduo_per_for_ts_3/detection/data/ --load_model ../exp/multi_pose/dla_3x/model_last.pth

# 3d
# 训练refinenet
CUDA_VISIBLE_DEVICES=4 python test_3d_train.py multi_pose_3d --exp_id dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10 --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth  --split val --vis_thresh 0.3 --debug 2 --K 20 --refine_loss l1 --match_max_center_dis 10.0
# 验证
CUDA_VISIBLE_DEVICES=6 python test_3d_val.py multi_pose_3d --exp_id dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10_val --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --refine_model_dir "/data1/yangfan/CenterNetExp/multi_pose_3d/dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10/model_Refine3d_38_4.651825075710478.pth" --split val --vis_thresh 0.3 --debug 2 --K 20 --refine_loss l1 --match_max_center_dis 10.0 --object_threshold 0.3
# 推断
CUDA_VISIBLE_DEVICES=2 python test_3d_val.py multi_pose_3d --exp_id seq3 --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --refine_model_dir "/data1/yangfan/CenterNetExp/multi_pose_3d/dla_3x_3d_0522_1231_fixCenterNet_RefineEasy_repairGT_l1loss_limit10/model_Refine3d_38_4.651825075710478.pth" --split val --vis_thresh 0.3 --debug 2 --K 20 --refine_loss l1 --match_max_center_dis 10.0 --object_threshold 0.3 --img_nogt_dir /data1/yangfan/HOLODataset/20210104/0114_3D数据标注前视_08_seq3/img/