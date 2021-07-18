import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

import proto.f3net as f3

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
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
        elif isinstance(m, (nn.Softmax, nn.ReLU)):
            pass
        else:
            m.initialize()

''' Contrast feature map '''
class CFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

''' Channel based relation '''
class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_up, enc_dim=128):
        super(CA, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, enc_dim, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(enc_dim)
        self.conv1  = nn.Conv2d(in_channel_up, enc_dim, 1, 1, 0)
        self.bn1    = nn.BatchNorm2d(enc_dim)
        self.conv2  = nn.Conv2d(enc_dim, enc_dim, 1, 1, 0)

    def forward(self, left, up):
        left= F.relu(self.bn0(self.conv0(left)), inplace=True)  
        up  = up.mean(dim=(2,3), keepdim=True)
        up  = F.relu(self.bn1(self.conv1(up)), inplace=True)
        up  = torch.sigmoid(self.conv2(up))
        return left * up

    def initialize(self):
        weight_init(self)

""" Relational Attention Module """
class RAM(nn.Module):
    def __init__(self, in_channels):
        super(RAM, self).__init__()    
        hidden_dim  = in_channels // 8
        self.conv0  = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv1  = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2  = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax= nn.Softmax(dim=-1)
        self.scale  = hidden_dim ** -0.5
    
    def forward(self, left, up):
        assert left.size() == up.size()
        batch_size, _, height, width = left.size()
        feat_key    = self.conv0(left).view(batch_size, -1, height * width).permute(0,2,1)
        feat_query  = self.conv1(up).view(batch_size, -1, height * width)
        attention   = self.softmax(torch.bmm(feat_key, feat_query) * self.scale)
        feat_val    = self.conv2(left).view(batch_size, -1, height * width)
        out = torch.bmm(feat_val, attention).view(batch_size, -1, height, width)

        return out
    
    def initialize(self):
        weight_init(self)

""" Attention Map Decoder """
class ChannelPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=1).unsqueeze(1)
    def initialize(self):
        return

class AMD(nn.Module):
    def __init__(self, in_channel) -> None:
        super(AMD, self).__init__()
        self.conv0  = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.bn0    = nn.BatchNorm2d(in_channel)
        self.pool   = ChannelPool()

    def forward(self, x):
        x   = F.relu(self.bn0(self.conv0(x)), inplace=True)
        x   = self.pool(x)
        return x

    def initialize(self):
        weight_init(self)


""" Refine attention map """
class RFM(nn.Module):
    def __init__(self, in_channel_left, in_channel_up, repr_dim=128) -> None:
        super(RFM, self).__init__()
        self.conv0  = nn.Conv2d(in_channel_left, 2*repr_dim, 3, 1, 1)
        self.bn0    = nn.BatchNorm2d(2*repr_dim)
        self.conv1  = nn.Conv2d(in_channel_up, repr_dim, 3, 1, 1)
        self.bn1    = nn.BatchNorm2d(repr_dim)
        self.conv2  = nn.Conv2d(repr_dim, repr_dim, 3, 1, 1)
        self.bn2    = nn.BatchNorm2d(repr_dim)
    
    def forward(self, left, up):
        '''left for coarse saliency map'''
        out1    = F.relu(self.bn0(self.conv0(left)), inplace=True)
        out2    = F.relu(self.bn1(self.conv1(up)), inplace=True)
        w, b    = out1[:, :128, :, :], out1[:, 128:, :, :]
        out     = F.relu(self.bn2(self.conv2(w * out2 + b)), inplace=True)
        return out
        
    def initialize(self):
        weight_init(self)

''' Supress-Augment Module '''
class SAM(nn.Module):
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
        '''left: previous predicted map'''
        left= F.relu(self.bn0(self.conv0(left)), inplace=True)
        up1  = F.relu(self.bn1(self.conv1(up)), inplace=True)
        prod= up1 * left
        # prod1= F.relu(self.bn2(self.conv2(prod)), inplace=True)
        # up  = up - prod1
        # prod2= F.relu(self.bn3(self.conv3(prod)), inplace=True)
        # out  = self.ram(up, prod2)

        return prod #out

    def initialize(self):
        weight_init(self)


'''Encoder-decoder arch'''
class Encoder(nn.Module):
    def __init__(self, cfg=None, enc_dim=128) -> None:
        super().__init__()
        self.cfg= cfg
        self.f3 = f3.F3Net(cfg)
        self.linear = nn.Sequential(
            nn.Conv2d(64*4, enc_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(enc_dim),
            nn.ReLU(inplace=True),
        )
        self.ca = CA(enc_dim, enc_dim)
        # self.ram= RAM(enc_dim)


    def forward(self, X):
        # pred, out2h, out3h, out4h, out5v
        outs    = list(self.f3(X))
        shape   = X.size()[2:]   
        
        for i in range(len(outs)):
            outs[i] = F.interpolate(outs[i], size=shape, mode='bilinear')
        out = torch.cat(outs, dim=1)
        out = self.linear(out)
        out1 = self.ca(out, out)
        # out2 = self.ram(out1, out1)
        return out + out1 #+ out2
    
    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self, enc_dim=128) -> None:
        super().__init__()
        self.amd1   = AMD(enc_dim)
        self.rfm    = RFM(1, enc_dim, enc_dim)
        self.amd2   = AMD(enc_dim)
        self.sam    = SAM(1, enc_dim)

    def mode(self, s):
        self.mode   = s

    def init_state(self, state):
        return state

    def forward(self, state, X):
        '''
        state: {batch_size, channel, h, w}
        X: {batch_size, num_step, h, w}
        to {num_step, batch_size, h, w}'''
        print('dec begin')
        X   = X.permute(1,0,2,3)
        dec_state   = state
        outs, dec_states = [], []
        for x in X:
            '''generate map'''
            feat= self.amd1(dec_state)
            feat= self.rfm(feat, dec_state)
            feat= self.amd2(feat)
            outs.append(feat)
            
            '''change state'''
            dec_state   = self.sam(feat, dec_state)
            # dec_states.append(dec_state)
        '''outs: len: num_step, (batch, 1, H, W)'''       
        return torch.cat(outs, dim=1), dec_state
    
    def initialize(self):
        weight_init(self)

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
        X: {batch_size, num_step, H, W} 
        valid_len: {batch_size}"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class net(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.encoder    = Encoder()
        self.decoder    = Decoder()
        self.cfg        = cfg
        self.initialize()
    
    def forward(self, enc_X, dec_X, valid_len):
        state = self.encoder(enc_X)
        state = self.decoder.init_state(state)
        dec_out, state  = self.decoder(state, dec_X)
        return sequence_mask(dec_out, valid_len)
    
    def initialize(self):
        weight_init(self)




