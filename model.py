import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import math
import torch.nn.functional as F
import time
from numba import cuda
import numpy as np
import itertools
#fea_n =1999

    

class MLPEncoder(nn.Module):
    def __init__(self, output_dim, fea_n):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim
        self.fea_n = fea_n

        self.fc1 = nn.Linear(fea_n+2, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 128)
        self.fc4 = nn.Linear(128, output_dim)

        self.act = nn.ReLU()
        self.m = nn.Dropout(p=0.2)

    def forward(self, x):
        h = x.view(-1, self.fea_n+2)
        h = self.m(self.act(self.fc1(h)))
        h = self.m(self.act(self.fc2(h)))
        h = self.m(self.act(self.fc3(h)))
        h = self.fc4(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder_C(nn.Module):
    def __init__(self, input_dim, fea_n):
        super(MLPDecoder_C, self).__init__()
        self.fea_n = fea_n
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 800),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(800, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(800, fea_n),
            
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        img_c = h.view(z.size(0), 1, self.fea_n)
        return img_c
    
    
class MLPDecoder_S(nn.Module):
    def __init__(self, input_dim, fea_n):
        super(MLPDecoder_S, self).__init__()
        self.fea_n = fea_n
        self.net = nn.Sequential(
            nn.Linear(128, 800),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(800, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(800, fea_n),
            
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        img_s = h.view(z.size(0), 1, self.fea_n)
        return img_s
    

class P_Layer(nn.Module):
    def __init__(self, input_dim,p_fea=128):
        super(P_Layer, self).__init__()
        
        self.p_fea = p_fea
        self.fc1 = nn.Linear(input_dim, p_fea)        
        self.act = nn.ReLU()
        self.m = nn.Dropout(p=0.2)

        

    def forward(self, z):
        h = z.view(z.size(0), -1)
        z_proj = self.act(self.fc1(h))
        z_proj = z_proj.view(z.size(0), 1, self.p_fea)
        return z_proj

    