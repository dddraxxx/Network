import torch
import torch.nn.functional as F
from torch import nn

####################################################################################################
# Fusion

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()


class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down1, in_channel_down2, in_channel_right):
        super(FAM, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down1, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channel_down2, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn3   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv_l  = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4   = nn.Conv2d(256 * 4, 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn4     = nn.BatchNorm2d(256)

    

    def forward(self, left, down0, down1, right):
        left  = F.relu(self.bn0(self.conv0(left)), inplace = True)  # 256 channels
        down0 = F.relu(self.bn1(self.conv1(down0)), inplace = True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace = True)  # 256
        down1 = F.relu(self.bn3(self.conv3(down1)), inplace = True)  # 256

        down_1 = self.conv_d1(down0)

        w1 = self.conv_l(left)
        if down0.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down0, size = left.size()[2:], mode = 'bilinear')
            z1 = F.relu(w1 * down_, inplace = True)
        else:
            z1 = F.relu(w1 * down0, inplace = True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size = left.size()[2:], mode = 'bilinear')

        z2 = F.relu(down_1 * left, inplace = True)

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size = left.size()[2:], mode = 'bilinear')
        z3 = F.relu(down_2 * left, inplace = True)

        down_3 = self.conv_d1(down1)
        if down_3.size()[2:] != left.size()[2:]:
            down_3 = F.interpolate(down_3, size = left.size()[2:], mode = 'bilinear')
        z4 = F.relu(down_3 * left, inplace = True)

        out = torch.cat((z1, z2, z3, z4), dim = 1)
        return F.relu(self.bn4(self.conv4(out)), inplace = True)

    def initialize(self):
        weight_init(self)

####################################################################################################
# Intuition Relation


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ObjectRelationModule(nn.Module):
    def __init__(self, planes, d1, d2):
        super(_ObjectRelationModule, self).__init__()
        self.inplanes  = planes
        self.outplanes = planes // 2

        self.conv_a1   = nn.Conv2d(self.inplanes, self.outplanes, kernel_size = 3, stride=1, padding=1, dilation=1)
        self.conv_a2   = nn.Conv2d(self.inplanes, self.outplanes, kernel_size = 3, stride=1, padding=d1, dilation=d1)
        self.bn1       = nn.BatchNorm2d(self.outplanes)

        self.conv_b1   = nn.Conv2d(self.inplanes, self.outplanes, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.conv_b2   = nn.Conv2d(self.inplanes, self.outplanes, kernel_size = 3, stride = 1, padding = d2, dilation = d2)
        self.bn2       = nn.BatchNorm2d(self.outplanes)

        self.relu = nn.ReLU()

        self.pam = _PositionAttentionModule(self.inplanes)

    def forward(self, x):
        conv_a1  = self.conv_a1(x)
        conv_a2  = self.conv_a2(x)
        object_1 = conv_a1 - conv_a2
        object_1 = self.bn1(object_1)
        object_1 = self.relu(object_1)

        conv_b1  = self.conv_b1(x)
        conv_b2  = self.conv_b2(x)
        object_2 = conv_b1 - conv_b2
        object_2 = self.bn2(object_2)
        object_2 = self.relu(object_2)

        output = self.pam(torch.cat((object_1, object_2), 1))

        return output


class _GestaltRelationModule(nn.Module):
    def __init__(self, planes, d1, d2):
        super(_GestaltRelationModule, self).__init__()
        self.inplanes    = planes
        self.interplanes = planes // 2
        self.outplanes   = planes // 4

        self.conv_a = nn.Sequential(
            nn.Conv2d(self.inplanes, self.interplanes, 3, 1, 1),
            nn.BatchNorm2d(self.interplanes), 
            nn.ReLU()
        )

        self.conv_b = nn.Sequential(
            nn.Conv2d(self.interplanes, self.outplanes, 3, 1, 1),
            nn.BatchNorm2d(self.outplanes), 
            nn.ReLU()
        )

        self.obm = _ObjectRelationModule(self.outplanes, d1, d2)
        self.pam = _PositionAttentionModule(self.inplanes)

    def forward(self, x):
        conv_a = self.conv_a(x)
        conv_b = self.conv_b(conv_a)

        gm_1 = self.obm(conv_b)
        gm_2 = self.obm(gm_1)
        gm_3 = self.obm(gm_2)
        gm_4 = self.obm(gm_3)

        output = self.pam(torch.cat((gm_1, gm_2, gm_3, gm_4), 1))

        return output

####################################################################################################
# Logical Relation


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta    = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _LogicalRelationModule(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(_LogicalRelationModule, self).__init__()
        interplanes = inplanes // 4
        self.conv_a = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding = 1, bias = False),
            nn.BatchNorm2d(interplanes), 
            nn.ReLU(inplace=False)
        )
        self.cam = _ChannelAttentionModule(interplanes)
        self.conv_b = nn.Sequential(
            nn.Conv2d(interplanes, outplanes, 3, padding = 1, bias = False),
            nn.BatchNorm2d(outplanes), 
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        output = self.conv_a(x)
        
        output = self.cam(output)
        output = self.cam(output)
        output = self.cam(output)
        output = self.cam(output)

        output = self.conv_b(output)

        return output

####################################################################################################
# Relation Module


class _RelationModule(nn.Module):
    def __init__(self, inplanes, outplanes, d1, d2, aux = True, norm_layer = nn.BatchNorm2d, norm_kwargs = None, **kwargs):
        super(_RelationModule, self).__init__()
        self.aux = aux
        interplanes = inplanes // 4

        self.conv_p1 = nn.Sequential(
            nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
            norm_layer(interplanes, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.conv_p2 = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
            norm_layer(interplanes, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.conv_c1 = nn.Sequential(
                    nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                    norm_layer(interplanes, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)
        )

        self.conv_c2 = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
            norm_layer(interplanes, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        self.grm = _GestaltRelationModule(interplanes, d1, d2)
        self.lrm = _LogicalRelationModule(inplanes, outplanes)

        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(interplanes, 1)
        )
        
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(interplanes, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(interplanes, 1)
            )

    def forward(self, x):
        feat_g = self.conv_p1(x)
        feat_g = self.grm(feat_g)
        feat_g = self.conv_p2(feat_g)

        feat_l = self.conv_c1(x)
        feat_l = self.lrm(feat_l)
        feat_l = self.conv_c2(feat_l)

        feat_fusion = feat_g + feat_l

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            g_out = self.conv_p3(feat_g)
            l_out = self.conv_c3(feat_l)
            outputs.append(g_out)
            outputs.append(l_out)

        return tuple(outputs)

####################################################################################################
# Net
class Net(nn.Module):
    def __init__(self, backbone_path = None):
        super(Net, self).__init__()
        resnet = ResNet50(backbone_path) #resnet50
        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.relation_4 = _RelationModule(2048, 2048, 2, 4)
        self.relation_3 = _RelationModule(1024, 1024, 4, 8)
        self.relation_2 = _RelationModule(512, 2048, 4, 8)
        self.relation_1 = _RelationModule(256, 256, 4, 8)
        
        self.feature = FAM(1024, 512, 256, 256)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        relation1 = self.relation_4(layer1)
        relation2 = self.relation_3(layer2)
        relation3 = self.relation_2(layer3)
        relation4 = self.relation_1(layer4)

        feature = self.feature(relation1, relation2, relation3, relation4)


