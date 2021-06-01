from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .holo3d import HoloDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'holo3d': HoloDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector, 
}
