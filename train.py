import datetime
import os
import sys
import torch

from torch.utils import data
from torch.utils.data.dataloader import DataLoader
sys.path.append('./')

import GeNet.Net as Net
import lib.train_util as t_util
import dataset
import torch.nn.functional as F



def train(dataset, network, datapath, savepath, **kargs):
    cfg     = t_util.Config(datapath=datapath, savepath=savepath,
        epoch=30, batch=16, weight_decay=5e-4, **kargs)
    cfg.mode= 'train'
    data    = dataset.Data(cfg)
    loader  = DataLoader(data, cfg.batch, shuffle=True, 
        num_workers=8, pin_memory=True)
    path    = os.path.join(cfg.savepath, datetime.datetime.now().strftime('%d-%H'))
    if not os.path.exists(path):
        os.makedirs(path)
    
    ### Network
    net     = network(cfg)
    net.train(True)
    net.cuda()

    ### Optim
    base, head  = [],[]
    for name, module in net.named_parameters():
        if 'bkbone' in name:
            base.append(module)
        else:
            head.append(module)
    
    optimizer   = torch.optim.SGD([{'params': base}, {'params': head}],
        weight_decay=cfg.weight_decay, lr=1e-3)
    ### Training
    global_step = 0
    total_step  = len(loader) * cfg.epoch
    lr_scheduler    = t_util.LR_Scheduler([1e-3, 0.1], total_step)
    
    for epoch in range(cfg.epoch):
        idx   = 0
        for img, mask, val_len in iter(loader):
            img     = img.to('cuda:0', torch.float32, non_blocking=True)
            mask    = mask.to('cuda:0', torch.float32, non_blocking=True)
            out     = net(img)
            loss    = F.binary_cross_entropy_with_logits(out, mask)

            idx   += 1
            global_step     += 1
            ### LR
            lr, momentum    = lr_scheduler.get_lr(idx)
            optimizer.momemtum  = momentum
            optimizer.param_groups[0]['lr']     = 0.1*lr

            optimizer.param_groups[1]['lr']     = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if idx%10==0:
                msg     = '%s | step: %d/%d, %d/%d, %d/%d | lr=%.5f | loss=%.6f' % (
                    datetime.datetime.now(), epoch+1, cfg.epoch, idx, len(loader), global_step, total_step, 
                    lr, loss.item()
                )
                print(msg)
        if (epoch+1) % 10 == 0:
            
            torch.save(net.state_dict(), path + '/model-%d'%(epoch+1))
            print('model saved to %s' % path)

    
if __name__=='__main__':
    train(dataset, Net.GeNet, './data/ASSR', 'output/', 
        rank_num=1, snapshot='output/14-02/model-29')
            