import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

''' Contrast feature map '''
class CFM(nn.module):
    def __init__(self) -> None:
        super().__init__()

''' Channel based relation '''
class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_up):
        super(CA, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, 256, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(256)
        self.conv1  = nn.Conv2d(in_channel_up, 256, 1, 1, 1)
        self.conv2  = nn.Conv2d(256, 256, 1, 1, 1)

    def forward(self, left, up):
        left= F.relu(self.bn0(self.conv0(left)), inplace=True)  
        up  = up.mean(dim=(2,3), keepdim=True)
        up  = F.relu(self.conv1(up), inplace=True)
        up  = torch.sigmoid(self.conv2(up))
        return left * up

""" Relational Attention module """
class RAM(nn.module):
    def __init__(self, in_channels):
        super(RAM, self).__init__()    
        self.conv0  = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv1  = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2  = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax= nn.Softmax(dim=-1)
    
    def forward(self, left, up):
        assert left.size() == up.size()
        batch_size, _, height, width = left.size()
        feat_key    = self.conv0(left).view(batch_size, -1, height * width).permute(0,2,1)
        feat_query  = self.conv1(up).view(batch_size, -1, height * width)
        attention   = self.softmax(torch.bmm(feat_key, feat_query))
        feat_val    = self.conv2(left).view(batch_size, -1, height * width)
        out = torch.bmm(feat_val, attention).view(batch_size, -1, height, width)

        return out

""" Attention Map Decoder """
class ChannelPool(nn.module):
    def forward(self, x):
        return x.mean(dim=1).unsqueeze(1)

class AMD(nn.module):
    def __init__(self, in_channel) -> None:
        super(AMD, self).__init__()
        self.conv0  = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(in_channel)
        self.pool   = ChannelPool()

    def forward(self, x):
        x   = F.relu(self.bn0(self.conv0(x)), inplace=True)
        x   = self.pool(x)
        return x


""" Refine attention map """
class RFM(nn.module):
    def __init__(self, in_channel_left, in_channel_up) -> None:
        super(RFM, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, 512, 3, 1, 1)
        self.bn0    = nn.BatchNorm2d(512)
        self.conv2  = nn.Conv2d(in_channel_up, 256, 3, 1, 1)
    
    def forward(self, left, up):
        out1    = F.relu(self.bn0(self.conv0(left)), inplace=True)
        out2    = self.conv2(up)
        w, b    = out1[:, :256, :, :], out1[:, 256:, :, :]
        return F.relu(w * out2 + b, inplace=True)

''' Supress-Augment module '''
class SAM(nn.module):
    def __init__(self, in_channel_left, in_channel_up) -> None:
        super(SAM, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, in_channel_up, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(in_channel_up)
        self.conv1  = nn.Conv2d(in_channel_up, in_channel_up, 3, 1, 1)
        self.bn1    = nn.BatchNorm2d(in_channel_up)
        self.conv2  = nn.Conv2d(in_channel_up, in_channel_up, 3, 1, 1)
        self.bn2    = nn.BatchNorm2d(in_channel_up)
        self.conv3  = nn.Conv2d(in_channel_up, in_channel_up, 3, 1, 1)
        self.bn3    = nn.BatchNorm2d(in_channel_up)
        self.ram    = RAM(in_channel_up)

    def forward(self, left, up):
        left= F.relu(self.bn0(self.conv0(left)), inplace=True)
        up  = F.relu(self.bn1(self.conv1(up)), inplace=True)
        prod= up * left
        prod1= F.relu(self.bn2(self.conv2(prod)), inplace=True)
        up  = up - prod1
        prod2= F.relu(self.bn3(self.conv3(prod)), inplace=True)
        out  = self.ram(prod2, up)

        return out


