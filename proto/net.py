import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

''' Channel based relation '''
class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_up):
        super(CA, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, 256, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(256)
        self.conv1  = nn.Conv2d(in_channel_up, 256, 1, 1, 1)
        self.conv2  = nn.Conv2d(256, 256, 1, 1, 1)

    def forward(self, left, up):
        left= F.relu(self.bn0(self.conv0(left)))  
        up  = up.mean(dim=(2,3), keepdim=True)
        up  = F.relu(self.conv1(up))
        up  = torch.tanh(self.conv2(up))
        return left * up + left

""" Relational Attention module """
class RAM(nn.module):
    def __init__(self, ):
        super().__init__()    
    
    def forward(self, left, up):
        assert left.size() == up.size()
        batch_size, _, height, width = left.size()
        feat_key    = self.conv0(left).view(batch_size, -1, height * width).permute(0,2,1)
        feat_query  = self.conv1(up).view(batch_size, -1, height * width)
        attention   = self.softmax(torch.bmm(feat_key, feat_query))
        feat_val    = self.conv2(left).view(batch_size, -1, height * width)
        out = torch.bmm(feat_val, attention).view(batch_size, -1, height, width)

        return out





# 

# Relational attention score
class 

# Refine map

# Contrast
