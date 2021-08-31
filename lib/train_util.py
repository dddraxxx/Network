from pickle import decode_long
import sys
import datetime

from lib.data_prefetcher import DataPrefetcher
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import dataset
from torch.utils.tensorboard import SummaryWriter


class LR_Scheduler:
    def __init__(self, lrs, total_steps, annealing_decay=1e-2, 
            momentums=[0.95, 0.85]) -> None:
        self.base_lr    = lrs[0]
        self.max_lr     = lrs[1]
        self.total_steps= total_steps
        self.annealing_decay    = annealing_decay
        self.momentums  = momentums
    
    def get_lr(self, cur):
        assert cur < self.total_steps
        x   = 1. - np.abs(cur*2./self.total_steps-1.)
        lr  = self.base_lr + (self.max_lr - self.base_lr)*x
        momentum = self.momentums[0] + (self.momentums[1] - self.momentums[0])*x
        return lr, momentum

class Config:
    def __init__(self, **kargs) -> None:
        self.kargs  = kargs
        self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
        self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
        print('\nParameters...')
        for k,v in kargs.items():
            print('%-10s: %s'%(k,v))
    
    def __getattr__(self, name):
        return self.kargs[name] if name in self.kargs else None


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/ASSR', savepath=SAVE_PATH, mode='train', 
        batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, 
        shuffle=True, num_workers=8)

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
    global_step = 0

    db_size = len(loader)
    for epoch in range(cfg.epoch):
        prefetcher  = DataPrefetcher(loader)
        batch = prefetcher.next()
        batch_idx   = -1

        while batch is not None:
        # for image, masks, valid_len in loader:
        #     image, masks, valid_len = image.cuda(), masks.cuda(), valid_len.cuda()
            image, masks, valid_len = batch
            niter   = epoch * db_size + batch_idx
            lr, momentum    = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter)
            optimizer.param_groups[0]['lr'] = 0.1 * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum  = momentum

            batch_idx   += 1
            global_step += 1

            outs    = net(image)
            loss    = F.binary_cross_entropy_with_logits(outs, masks[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item())
                print(msg)
            batch = prefetcher.next()

        if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath + '/model-%s'%(str(epoch+1)))

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

    
    

            


        


