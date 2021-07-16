from pickle import decode_long
import sys
import datetime

import proto.net as network
from lib.data_prefetcher import DataPrefetcher
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import dataset
from tensorboardX import SummaryWriter

SAVE_PATH = "./out"

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum

BASE_LR = 1e-3
MAX_LR  = 1e-1

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/home/qihuadong2/saliency_rank/GCPANet/data/ASSR', savepath=SAVE_PATH, mode='train', 
        batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
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
    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], 
        lr=0.05, momentum=0.9, 
        weight_decay=cfg.decay, nesterov=True)
    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)
    for epoch in range(cfg.epoch):
        prefetcher  = DataPrefetcher(loader)
        image, masks, valid_len = prefetcher.next()
        batch_idx   = -1

        while image is not None:
            niter   = epoch * db_size + batch_idx
            lr, momentum    = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch, niter)
            optimizer.param_groups[0]['lr'] = 0.1 * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum  = momentum

            batch_idx   += 1
            global_step += 1

            outs    = net(image, masks, valid_len)
            loss    = F.binary_cross_entropy_with_logits(outs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item())
                print(msg)
            image, masks, valid_len = prefetcher.next()

        if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath + '/model-%s'%(str(epoch+1)))

if __name__=='__main__':
    train(dataset, network.net)
    
    

            


        


