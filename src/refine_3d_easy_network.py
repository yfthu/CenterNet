import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
from util_3d import trucate_angle

class Refine_3d_easy_Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Refine_3d_easy_Network, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, input_size),
        )

    def forward(self, x):
        out = self.linear(x)
        for one_out in out:
            one_out[4] = trucate_angle(one_out[4])
        tmp = out + x
        for one_object in tmp:
            one_object[4] = trucate_angle(one_object[4])
        return tmp
