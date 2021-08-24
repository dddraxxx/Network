import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import numpy as np

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('res/resnet50-19c8e357.pth'), strict=False)


class ATM(nn.Module):
    def __init__(self, num_head, in_planes, num_dim):
        super().__init__()
        self.num_head   = num_head
        self.head_dim   = num_dim // num_head
        self.q  = nn.Conv2d(in_planes, num_dim, 1, 1, 0)
        self.k  = nn.Conv2d(in_planes, num_dim, 1, 1, 0)
        self.bn = nn.BatchNorm2d(num_head)
        # selective channel augmentation
        self.v  = nn.Conv2d(num_head, in_planes, 1, 1, 0)
  
    def forward(self, query, key, value):
        # query:(b,c,h,w) => qu:(b, num_head, h*w, q)
        B, _, H, W  = query.size()
        qu  = self.q(query).reshape(B, self.num_head, self.head_dim, H*W).permute(0, 1, 3, 2)
        ke  = self.k(key).reshape(B, self.num_head, self.head_dim, H*W)
        score   = F.relu(qu@ke).sum(dim=-1).reshape(B, self.num_head, H, W)
        score   = self.bn(score)
        score   = torch.sigmoid(self.v(score))
        return value * score

    def initialize(self):
        weight_init(self) 


class HAM(nn.Module):
    def __init__(self, num_dim, h_planes, l_planes, num_cardi=4):
        super().__init__()
        self.conv0  = nn.Conv2d(h_planes, num_dim, 3, 1, 1)
        self.conv2  = nn.Conv2d(l_planes, num_dim, 3, 1, 1)
        self.conv1  = nn.Conv2d(num_dim, num_dim, 3, 1, 1, groups=num_cardi)
        self.conv3  = nn.Conv2d(num_dim, num_dim, 3, 1, 1, groups=num_cardi)
        self.conv5  = nn.Conv2d(num_dim, num_dim, 1, 1, 0)
        self.conv4  = nn.Conv2d(num_dim, num_dim, 3, 1, 1)

        self.bn0    = nn.BatchNorm2d(num_dim)
        self.bn1    = nn.BatchNorm2d(num_dim)
        self.bn2    = nn.BatchNorm2d(num_dim)
        self.bn3    = nn.BatchNorm2d(num_dim)
        self.bn4    = nn.BatchNorm2d(num_dim)
        self.bn5    = nn.BatchNorm2d(num_dim)
    
    def forward(self, high, low):
        high    = F.relu(self.bn0(self.conv0(high)), inplace=True)
        low     = F.relu(self.bn2(self.conv2((low))), inplace=True)

        w_h     = F.gelu(self.bn1(self.conv1(F.interpolate(high, size=low.size()[2:],
            mode='bilinear'))))
        w_l     = F.relu(self.bn3(self.conv3(low)), inplace=True)
        out     = F.relu(self.bn5(self.conv5(w_h*w_l)), inplace=True)

        return F.relu(self.bn4(self.conv4(out + low)), inplace=True)
    
    def initialize(self):
        weight_init(self)
    

class GeNet(nn.Module):
    def __init__(self, cfg, img_size=None, rank_num=None):
        super().__init__()
        self.cfg    = cfg
        self.bkbone = ResNet()
        
        # self.pos_emb= nn.Parameter(torch.zeros((2, img_size, img_size)))
        self.atm1   = ATM(4, 256, 256)
        self.atm2   = ATM(4, 256, 256)
        self.ham1   = HAM(256, 256, 512, 4)
        self.ham2   = HAM(256, 256, 256, 4)
        self.conv5  = nn.Conv2d(2048, 256, 3, 1, 1)
        self.bn5    = nn.BatchNorm2d(256)
        self.conv4  = nn.Conv2d(1024, 256, 3, 1, 1)
        self.bn4    = nn.BatchNorm2d(256)
        self.conv1  = nn.Conv2d(256, 1, 3, 1, 1)
        self.initialize()
    
    def forward(self, x):
        outs    = (self.bkbone(x))
        out2, out3, out4, out5 = outs
        out5b   = F.relu(self.bn5(self.conv5(out5)), inplace=True)
        out5a   = self.atm1(out5b, out5b, out5b)
        out4    = F.relu(self.bn4(self.conv4(out4)), inplace=True)
        out5a   = F.interpolate(out5a, size=out4.shape[2:], mode='bilinear')
        out4a   = self.atm2(out5a, out4, out4)
        out3a   = self.ham1(out4a, out3)
        out2a   = self.ham2(out3a, out2)
        out     = F.interpolate(self.conv1(out2a), size=x.shape[2:],
            mode='bilinear')
        return out
    
    def initialize(self):
        if self.cfg.snapshot:
            print('loading snapshot model')
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)




