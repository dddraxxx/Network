import os
import sys
sys.path.append('/home/crh/saliency_rank/Network')

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import dataset
import GeNet.Net as Net
from scipy import stats

def cal_SOR(masks, mask, IOU_thres=0.5, sal_thres=0.1):
    so  = mask.unique(sorted=True).flip(dims=[0])
    mask= mask.squeeze()
    srt = []
    for o in so[:-1]:
        loc = (mask == o)
        IOU = torch.where(loc, masks, torch.zeros_like(masks).cuda())
        IOU = (IOU > sal_thres).sum() / loc.sum()
        if IOU > IOU_thres:
            srt += [[o, masks[loc].mean()]]

    if len(so)<=2 and len(srt):
        return 1, 1
    srt = torch.tensor(srt).cpu().numpy()
    cor = 0
    if len(srt):
        cor, p = stats.spearmanr(srt[:,0], srt[:,1])
    return len(srt), cor


class Test():
    def __init__(self, cfg, network):
        self.cfg    = cfg
        self.data   = dataset.Data(cfg)
        self.dataloader     =  DataLoader(self.data, batch_size=1, num_workers=3)
        self.net    = network(cfg)

    def eval(self, save=True):
        net = self.net
        net.cuda()
        net.eval()
        t_mae = 0
        t_sor = 0
        l = len(self.dataloader)
        num_img_used = 0
        with torch.no_grad():
            for img, mask, shape, name in self.dataloader:
                name        = name[0]
                img, mask   = img.cuda().float(), mask.cuda().float()
                ## predict
                masks       = net(img)
                pred        = torch.max(masks, dim=1).values
                pred        = torch.sigmoid(pred)
                pred        = TF.resize(pred, size=shape).squeeze()
                ## mae
                mae         = (pred-mask).abs().mean()
                t_mae       += mae
                ## sor
                num, sor    = cal_SOR(pred, mask)
                if not (np.isnan(sor) or num==0):
                    num_img_used += 1
                    t_sor += sor
                # t_sor       += sor 
                print(f'id={name} mae={mae:.6f} sor={f"{sor:.3f}":<7} obj_in_common={num:<4d} img_used={num_img_used}')
                ## save
                if save:
                    path    = os.path.join(cfg.savepath, name+'.png')
                    pred    = np.uint8((pred*255).cpu().numpy())
                    if not os.path.exists(cfg.savepath):
                        os.makedirs(cfg.savepath)
                    cv2.imwrite(path, pred)
        print(f'summary: mae={t_mae/l} sor={t_sor/num_img_used}')

if __name__ == '__main__':
    from lib.train_util import Config
    cfg = Config(rank_num=3, snapshot=sys.argv[1], mode='test',
         datapath='./data/ASSR',
         savepath = './pred_maps')
    t = Test(cfg, Net.GeNet)
    t.eval()