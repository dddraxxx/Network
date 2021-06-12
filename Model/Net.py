import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnet import resnet50

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