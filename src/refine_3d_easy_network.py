import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F


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
        return out + x
