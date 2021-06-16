import os
import cv2

# image path
#im_dir = '/data1/lvmengyao/RTM3DExp/523_RTM3D_dla34_debug/results/VThresh0.3/'
im_dir = '/data1/yangfan/CenterNetExp/multi_pose_3d/dla_3x_3d_limit10_seq3_val/val/epoch130/'
#im_dir = '/data1/lvmengyao/RTM3DExp/325_RTM3D_dla34_504sample/results/0114_3D数据标注前视_08_seq3/VThresh0.3/'
# output video path
video_dir = './seqvideos/'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
# set saved fps
fps = 10
# get frames list
frames = sorted(os.listdir(im_dir))
#frames = frames[6885-5759:7454-5759]
frames = frames[453:453+250]
# w,h of image
img = cv2.imread(os.path.join(im_dir, frames[0]))
img_size = (img.shape[1], img.shape[0])
# get seq name
seq_name = os.path.dirname(im_dir).split('/')[-1]
# splice video_dir
video_dir = os.path.join(video_dir, seq_name+'_seq3_0606' + '.avi')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames:
    f_path = os.path.join(im_dir, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + " has been written!")

videowriter.release()
