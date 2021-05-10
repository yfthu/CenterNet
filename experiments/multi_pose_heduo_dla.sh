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
CUDA_VISIBLE_DEVICES=0 python demo.py multi_pose --exp_id dla_1x_2 --demo ../data/heduo/test/ --load_model ../exp/multi_pose/dla_1x_2/model_best.pth