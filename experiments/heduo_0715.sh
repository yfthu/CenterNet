python main.py multi_pose --exp_id dla_1x_3 --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 3,4 --num_workers 8 --master_batch 2 --debug 5 --display_env  dla_1x_3 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --display_port 8098


python main.py multi_pose --arch res_18 --head_conv 64 --exp_id res_18_14kps --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 4,5,6 --num_workers 8 --master_batch 2 --debug 5 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --add_kps

python main.py multi_pose --arch res_18 --head_conv 64 --exp_id res_18_14kps --dataset coco_hp --batch_size 8 --lr 3.12e-5 --gpus 3,4 --num_workers 8 --master_batch 2 --debug 5 --num_epochs 300 --lr_step 40,80,120,240 --K 40 --add_kps