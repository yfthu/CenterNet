# train
CUDA_VISIBLE_DEVICES=4,5,6 python main.py multi_pose --exp_id dla_1x_3 --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 0,1,2 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_1x_3 --num_epochs 300 --lr_step 40,80,120,240 --K 40
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multi_pose --exp_id dla_3x --dataset coco_hp --batch_size 11 --lr 4.3e-5 --gpus 0,1,2,3 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_3x --num_epochs 300 --lr_step 60,120,240 --K 40

# val, prepare coco test json file using convert script, save results.json
# --resume automatically load the latest ckpt if load_model is empty
# no GT
CUDA_VISIBLE_DEVICES=0 python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --resume --split test2 --vis_thresh 0.3 --debug 1
# also save GT for comparison
CUDA_VISIBLE_DEVICES=0 python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_1x_2/model_best.pth --split test2 --vis_thresh 0.3 --debug 2

# test, no GT and no results.json saved, only save predicted images
CUDA_VISIBLEVICES=0 python demo.py multi_pose --exp_id dla_1x_2 --demo ../data/heduo/test/ --load_model ../exp/multi_pose/dla_1x_2/model_best.pth


#ziji:

# train
CUDA_VISIBLE_DEVICES=4,5,6 python main.py multi_pose --exp_id dla_1x_3 --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 0,1,2 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_1x_3 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --display_port 8098
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py multi_pose --exp_id dla_3x --dataset coco_hp --batch_size 11 --lr 4.3e-5 --gpus 0,1,2,3 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_3x --num_epochs 300 --lr_step 60,120,240 --K 40 --display_port 8098



# also save GT for comparison
CUDA_VISIBLE_DEVICES=0 python test.py multi_pose --exp_id dla_1x_14kps_0512_2 --dataset coco_hp --keep_res --load_model ../exp_lmy/multi_pose/dla_1x_14kps/model_best.pth --split val --vis_thresh 0.3 --debug 2

# test, no GT and no results.json saved, only save predicted images
CUDA_VISIBLEVICES=0 python demo.py multi_pose --exp_id dla_1x_2 --demo ../data/heduo/test/ --load_model ../exp_lmy/multi_pose/dla_1x_2/model_best.pth







#0512:
"/data1/yangfan/CenterNetExp/multi_pose/dla_3x/model_last.pth"
CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --split val --vis_thresh 0.3 --debug 2

#fei
CUDA_VISIBLE_DEVICES=0 python test.py multi_pose --exp_id fei_dla_3x --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --split val --vis_thresh 0.3 --debug 2
CUDA_VISIBLE_DEVICES=0 python main.py multi_pose --exp_id fei_dla_3x --dataset coco_hp --batch_size 2 --lr 4.3e-5 --gpus 0 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_3x --num_epochs 300 --lr_step 60,120,240 --K 40 --load_model ../exp/multi_pose/dla_3x/model_last.pth

#0513:
CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id 0513_dla_3x_hm_score_factor --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --split val --vis_thresh 0.3 --debug 2

#test2
CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id 0513_dla_3x_hm_score_factor --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --split test2 --vis_thresh 0.3 --debug 1
CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id dla_3x --dataset coco_hp --keep_res --load_model ../exp/multi_pose/dla_3x/model_last.pth --split test2 --vis_thresh 0.3 --debug 1


#0515
CUDA_VISIBLE_DEVICES=2,3,4,5,6 python main.py multi_pose --exp_id hg_1x_heduo --dataset coco_hp --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --load_model ../models/multi_pose_hg_1x.pth --gpus 0,1,2,3,4 --num_epochs 50 --lr_step 40 --K 40 --debug 5 --display_env hg_1x_heduo --display_port 8098

# 0515 10.08 跑起来了：pid 30796
CUDA_VISIBLE_DEVICES=2,3,4,5,6 python main.py multi_pose --exp_id hg_1x_heduo --dataset coco_hp --arch hourglass --batch_size 18 --master_batch 2 --lr 1.875e-4 --load_model ../models/multi_pose_hg_1x.pth --gpus 0,1,2,3,4 --num_epochs 50 --lr_step 40 --K 40 --input_h 480 --input_w 640 --debug 5 --display_env hg_1x_heduo --display_port 8098

CUDA_VISIBLE_DEVICES=3 python test.py multi_pose --exp_id hg_1x_heduo --dataset coco_hp --keep_res --load_model ../exp/multi_pose/hg_1x_heduo/model_last.pth --split val --vis_thresh 0.3 --debug 2 --input_h 480 --input_w 640




# main3d: fei
#fei python main_3d.py multi_pose --exp_id fei_dla_3x --dataset coco_hp --batch_size 2 --lr 4.3e-5 --gpus 0 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_3x --num_epochs 300 --lr_step 60,120,240 --K 40 --keep_res --split val --load_model ../exp/multi_pose/dla_3x/model_last.pth