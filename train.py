import sys
import datetime

import proto.net as net
from lib.data_prefetcher import DataPrefetcher
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import dataset
from tensorboardX import SummaryWriter

SAVE_PATH = ""

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/MSRA-B', savepath=SAVE_PATH, mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)
    for epoch in range(cfg.epoch):
        prefetcher  = DataPrefetcher(loader)
        image, masks, valid_len = prefetcher.next()
        batch_idx   = -1

        while image is not None:
            niter   = epoch * db_size + batch_idx
            lr, momentum    = []
            # optimizer. = 
            batch_idx   += 1
            global_step += 1

            outs    = net(image, masks, valid_len)
            loss    = F.binary_cross_entropy_with_logits(outs, masks[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            image, masks, valid_len = prefetcher.next()

        if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath + 'model-%s'%(str(epoch+1)))

if __name__=='__main__':
    train(dataset, net)
            


        


