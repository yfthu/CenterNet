# train, possible splits [523, 325, 217]
# centernet
python main.py holo3d --exp_id 523_dla --dataset holo --holo_split 523 --K 100 --batch_size 16 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0,1 --num_epochs 140 --lr 1.25e-4
python main.py holo3d --exp_id 523_dla --dataset holo --holo_split 523 --K 100 --batch_size 16 --master_batch 4 --num_epochs 70 --lr_step 45,60 --gpus 0,1,2 --num_epochs 140 --lr 1.25e-4
python main.py holo3d --exp_id 523_dla --dataset holo --holo_split 523 --K 100 --batch_size 25 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 2,3,4 --num_epochs 140 --lr 1.95e-4

# test
python test.py holo3d --exp_id 523_dla --dataset holo --holo_split 523 --resume