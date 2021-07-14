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

def get_instance_masks_by_ranks(map):
    rank_vals = (torch.sort(torch.unique(map), descending=True)[:-1])[0]
    masks=torch.stack([torch.where(map==val, torch.tensor(1.), torch.tensor(0.)
        ) for val in rank_vals], dim=0)
    return masks


path='/home/qihuadong2/saliency_rank/Attention_Shift_Ranks/'\
    + 'data/ASSR/gt/test/COCO_val2014_000000000192.png'
map=cv2.imread(path,0)
img=get_instance_masks_by_ranks(torch.tensor(map))
print(img.shape, img.dtype)

# for i in img:
#     print(np.unique(i))
#     fig, ax = plt.subplots()
#     ax.imshow(i,cmap='gray')
# plt.show()
SAVE_PATH = ""



def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/MSRA-B', savepath=SAVE_PATH, mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
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
        batch_idx   = -1

        while image is not None:
            image, mask = prefetcher.next()

            niter   = epoch * db_size + batch_idx
            lr, momentum    = []
            # optimizer. = 
            batch_idx   += 1
            global_step += 1

            masks   = get_instance_masks_by_ranks(mask)
            outs    = net(image, masks)
            loss    = 0
            for i, out in enumerate(outs):
                loss    += F.binary_cross_entropy_with_logits(out, masks[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath + 'model-%s'%(str(epoch+1)))

if __name__=='__main__':
    train(dataset, net)
            


        


