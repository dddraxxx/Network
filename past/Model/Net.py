import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnet import resnet50

class RAttention(nn.Module):
    '''This part of code is refactored based on https://github.com/Serge-weihao/CCNet-Pure-Pytorch. 
       We would like to thank Serge-weihao and the authors of CCNet for their clear implementation.'''
    def __init__(self,in_dim):
        super(RAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_query_LR = torch.diagonal(proj_query, 0, 2, 3)
        proj_query_RL = torch.diagonal(torch.transpose(proj_query, 2, 3), 0, 2, 3)
        # .contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_key_LR = torch.diagonal(proj_key, 0, 2, 3).permute(0,2,1).contiguous()
        proj_key_RL = torch.diagonal(torch.transpose(proj_key, 2, 3), 0, 2, 3).permute(0,2,1).contiguous()

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        proj_value_LR = torch.diagonal(proj_value, 0, 2, 3)
        proj_value_RL = torch.diagonal(torch.transpose(proj_value, 2, 3), 0, 2, 3)

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)

        # energy_LR = torch.bmm(proj_query_LR, proj_key_LR)
        # energy_RL = torch.bmm(proj_query_RL, proj_key_RL)
        energy_LR = torch.bmm(proj_key_LR, proj_query_LR)
        energy_RL = torch.bmm(proj_key_RL, proj_query_RL)


        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        out_LR = self.softmax(torch.bmm(proj_value_LR, energy_LR).unsqueeze(-1))
        out_RL = self.softmax(torch.bmm(proj_value_RL, energy_RL).unsqueeze(-1))

        # print(out_H.size())
        # print(out_LR.size())
        # print(out_RL.size())


        return self.gamma*(out_H + out_W + out_LR + out_RL) + x

class Relation_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Relation_Attention, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.ra = RAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))

            
    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.ra(output)
        output = self.convb(output)
        
        return output

############################ NETWORK ##############################
class NET(nn.Module):
    def __init__(self, backbone_path=None):
        super(NET, self).__init__()
        resnet = resnet50(backbone_path)
        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.edge_extract = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64))
        
        self.edge_predict = nn.Sequential(nn.Conv2d(64+512, 1, 3, 1, 1))

        self.contrast_4 = Contrast_Module_Deep(2048,d1=2, d2=4) # 2048x 12x12
        self.contrast_3 = Contrast_Module_Deep(1024,d1=4, d2=8) # 1024x 24x24
        self.contrast_2 = Contrast_Module_Deep(512, d1=4, d2=8) # 512x 48x48
        self.contrast_1 = Contrast_Module_Deep(256, d1=4, d2=8) # 256x 96x96

        self.ra_4 = Relation_Attention(2048, 2048)
        self.ra_3 = Relation_Attention(1024, 1024)
        self.ra_2 = Relation_Attention(512, 512)
        self.ra_1 = Relation_Attention(256, 256)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up_1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(256)
        self.cbam_2 = CBAM(128)
        self.cbam_1 = CBAM(64)

        self.layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)

        # self.refinement = Refinement_Net(1+1+3)
        # self.refinement = RCCAModule(1+1+3, 512, 1)
        self.refinement = nn.Conv2d(1+1+3+1+1+1, 1, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        contrast_4 = self.contrast_4(layer4)
        cc_att_map_4 = self.ra_4(layer4)
        final_contrast_4 = contrast_4 * cc_att_map_4

        # final_contrast_4 = torch.cat((contrast_4, cc_att_map_4), 1)

        up_4 = self.up_4(final_contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)
        layer4_map = F.sigmoid(layer4_predict)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        cc_att_map_3 = self.ra_3(layer3 * layer4_map)

        # final_contrast_3 = torch.cat((contrast_3, cc_att_map_3), 1)
        final_contrast_3 = contrast_3 * cc_att_map_3

        up_3 = self.up_3(final_contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)
        layer3_map = F.sigmoid(layer3_predict)

        contrast_2 = self.contrast_2(layer2 * layer3_map)
        cc_att_map_2 = self.ra_2(layer2 * layer3_map)
        # final_contrast_2 = torch.cat((contrast_2, cc_att_map_2), 1)
        final_contrast_2 = contrast_2 * cc_att_map_2

        up_2 = self.up_2(final_contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)
        layer2_map = F.sigmoid(layer2_predict)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        cc_att_map_1 = self.ra_1(layer1 * layer2_map)
        # print(cc_att_map_1)
        final_contrast_1 = contrast_1 * cc_att_map_1

        # final_contrast_1 = torch.cat((contrast_1, cc_att_map_1), 1)

        up_1 = self.up_1(final_contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        edge_feature = self.edge_extract(layer1)
        layer4_edge_feature = F.upsample(cbam_4, size=edge_feature.size()[2:], mode='bilinear', align_corners=True)
        
        final_edge_feature = torch.cat( (edge_feature, layer4_edge_feature), 1)
        
        layer0_edge = self.edge_predict(final_edge_feature)

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        layer0_edge = F.upsample(layer0_edge, size=x.size()[2:], mode='bilinear', align_corners=True)

        final_features = torch.cat((x, layer1_predict, layer0_edge, layer2_predict, layer3_predict, layer4_predict),1)
        final_predict = self.refinement(final_features)
        final_predict = F.upsample(final_predict, size=x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_edge, final_predict

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict), F.sigmoid(layer0_edge), F.sigmoid(final_predict)