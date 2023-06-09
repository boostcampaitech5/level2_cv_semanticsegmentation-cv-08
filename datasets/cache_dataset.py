from torch.utils.data import Dataset
import os
import pickle
import gzip
import numpy as np
import torch
from albumentations import (
    Compose,
    Normalize,
    Rotate,
)

class CacheDataset(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        super().__init__()
        
        if is_train:
            self.datadir = config.train_cache_data_dir
            self._filename = [n for n in os.listdir(self.datadir) if n.endswith('pkl')]
        else:
            self.datadir = config.valid_cache_data_dir
            self._filename = [n for n in os.listdir(self.datadir) if n.endswith('pkl')]

        self.transforms = transforms
        
    def __getitem__(self, idx):
        with gzip.open(os.path.join(self.datadir, self._filename[idx]), mode='rb') as f:
            p = pickle.load(f)

        if self.transforms is not None:
            # inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            image, label = np.array(p[0]).transpose(1, 2, 0), np.array(p[1]).transpose(1, 2, 0)
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result['mask']

            image = image.transpose(2, 0, 1)   
            label = label.transpose(2, 0, 1)
            
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            return image, label            

        return p[0], p[1]
    
    def __len__(self):
        return len(self._filename)

    

if __name__ == "__main__":
    import time
    transforms = Compose(
        [
            Rotate(limit=15),
            Normalize(mean=0.12099, std=0.16470, max_pixel_value=1., p=1.0),
        ])
    dataset = CacheDataset(None,
                           is_train=True, transforms=None)
    
    st = time.time()
    image, label = dataset[0]
    ed = time.time()
    print(image.shape, label.shape, f"time : {(ed-st) * 1000} ms")
    # torch.Size([3, 1024, 1024]) torch.Size([29, 1024, 1024]) time : 1069.817304611206 ms

    dataset = CacheDataset(None,
                           is_train=True, transforms=None)
    st = time.time()
    image, label = dataset[0]
    ed = time.time()
    print(image.shape, label.shape, f"time : {(ed-st) * 1000} ms")
    # torch.Size([3, 1024, 1024]) torch.Size([29, 1024, 1024]) time : 623.7030029296875 ms