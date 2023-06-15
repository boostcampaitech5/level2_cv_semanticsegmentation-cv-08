import gzip
import os
import pickle
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.transforms = transforms
        
        if self.is_train:
            self.data_dir = config.train_cache_data_dir
            self._filename = glob(os.path.join(self.data_dir, '*.pkl'))
        else:
            self.data_dir = config.valid_cache_data_dir
            self._filename = glob(os.path.join(self.data_dir, '*.pkl'))
        
        assert len(self._filename) != 0, "Please check cache data directory !"
    
    def __len__(self):
        return len(self._filename)

    def __getitem__(self, idx):
        with gzip.open(os.path.join(self.data_dir, self._filename[idx]), mode="rb") as f:
            image, label = pickle.load(f)

        if self.transforms is not None:
            # (c, h ,w) -> (h, w, c)
            image = np.array(image).transpose(1, 2, 0)
            label = np.array(label).transpose(1, 2, 0)

            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"]

            # (h, w, c) -> (c, h ,w)
            image = image.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)

            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            return image, label

        return image, label


