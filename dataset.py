#%%
from matplotlib.pyplot import axis
from pathlib import Path
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask

class RandomCrop:
    def __init__(self, H, W) -> None:
        self.H  = H
        self.W  = W
        
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = H - self.H
        randh   = W - self.W
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[:, p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:].copy(), mask[:, :, ::-1].copy()
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, masks):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        masks  = np.stack([
            cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            for mask in masks
        ], axis=0)
        return image, masks

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask

class Compose:
    def __init__(self, *ops) -> None:
        self.ops = ops
    
    def __call__(self, image, mask):
        for op in self.ops:
            image, mask= op(image, mask)
        return image, mask

###########################  Dataset Configuration  ############################
class Data(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg    = cfg
        self.rank_num   = cfg.rank_num if cfg.rank_num else 1
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop(288, 288)
        self.randomflip = RandomFlip()
        self.resize     = Resize(320, 320)
        self.totensor   = ToTensor()
        txtpath = Path(cfg.datapath).joinpath(cfg.mode + '.txt')

        with open(txtpath, 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) :
        name  = self.samples[idx]
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg').astype(np.float32)[:,:,::-1]
        mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)
        shape = mask.shape

        if self.cfg.mode=='train':
            mask, val_len = get_instance_masks_by_ranks(mask, 
                self.rank_num, trim=True)
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, val_len
        else:
            image, mask = self.normalize(image, mask)
            image, _ = self.resize(image, np.ones(mask[None].shape))
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

def trimit(maps, length):
    valid_len   = min(length, len(maps))
    mask_len    = length - valid_len
    mask        = np.tile(np.zeros(maps[0].shape),(mask_len, 1, 1))
    return np.concatenate([maps[:valid_len], mask], axis=0)

def get_instance_masks_by_ranks(map, num=1, trim=False):
    rank_vals = np.unique(map)[1:][::-1]
    r_num = min(num, len(rank_vals))
    masks= np.stack([(map == val).astype(np.float32)*255
     for val in rank_vals[:r_num]], axis=0)
    if trim and r_num<num:
        return trimit(masks, num), r_num
    return masks, r_num



if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    from lib.train_util import *
    cfg  = Config(mode='train', datapath='/home/crh/saliency_rank/Network/data/ASSR',
         rank_num=3)
    data = Data(cfg)
    plt.figure(figsize=(5,5))
    for i in range(1000):
        image, mask, a = data[i]
        image       = image.permute(1, 2, 0)*cfg.std + cfg.mean
        mask        = mask.squeeze(0)
        plt.subplot(141)
        plt.imshow(np.uint8(image))
        for i in range(3):
            plt.subplot(1,4,i+2)    
            plt.imshow(mask[i], cmap='gray')
        plt.show()
        a=input()
        if a=='!':
            break