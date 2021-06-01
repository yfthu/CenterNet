from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .holo3d import HoloTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer,
  'ddd': DddTrainer,
  'holo3d': HoloTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
}
