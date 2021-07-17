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

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg    = cfg
        self.rank_num   = cfg.rank if cfg.rank else 6
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(352, 352)
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
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.cfg.datapath+'/mask/' +name+'.png', 0).astype(np.float32)
        shape = mask.shape

        if self.cfg.mode=='train':
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            return image, mask
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]
        valid_len   = []
        for i in range(len(batch)):
            print('Calc 1 img-msk pair')
            mask[i] = get_instance_masks_by_ranks(mask[i])
            valid_len.append(len(mask[i]))
            image[i]= cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = [cv2.resize(m,  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
                for m in mask[i]]

            mask[i] = trim(mask[i], self.rank_num + 1)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2).float()
        mask  = torch.from_numpy(np.stack(mask, axis=0)).float()
        valid_len   = torch.tensor(valid_len).int()
        print(image.size(), mask.size(), valid_len.size())

        return image, mask, valid_len 

def trim(maps, length):
    valid_len   = min(length, len(maps))
    mask_len    = length - valid_len
    mask        = np.tile(np.zeros(maps[0].shape),(mask_len, 1, 1))
    return np.concatenate([maps[:valid_len], mask]
    , axis=0)

def get_instance_masks_by_ranks(map):
    rank_vals = np.sort(np.unique(map))[::-1]
    masks= np.array([(map == val).astype(np.float32)
     for val in rank_vals])
    return masks



if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='/home/qihuadong2/'
        +'saliency_rank/F3Net/data/ASSR')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        print(np.unique(mask))
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()