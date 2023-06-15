# python native
import os
from glob import glob

# external library
import h5py
import numpy as np

# torch
import torch
from torch.utils.data import Dataset


class Hdf5Dataset(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        self.config = config
        self.is_train = is_train
        self.transforms = transforms
        self.file = None

        if self.is_train:
            self.data_dir = config.train_hdf5_data_dir
            self._filename = glob(os.path.join(self.data_dir, "*.h5py"))[0]
        else:
            self.data_dir = config.valid_hdf5_data_dir
            self._filename = glob(os.path.join(self.data_dir, "*.h5py"))[0]
        
        assert len(self._filename) != 0, "Please check hdf5 data directory !"

    def __len__(self):
        with h5py.File(self._filename, "r") as hf:
            return len(hf.keys())

    def __getitem__(self, idx):
        self.file = h5py.File(self._filename, "r")

        image = self.file[str(idx)]["image"]
        label = self.file[str(idx)]["label"]

        # uint8 -> float32
        image = (np.array(image) / 255.0).astype(np.float32)
        label = (np.array(label) / 255.0).astype(np.float32)

        if self.transforms is not None:
            # (c, h ,w) -> (h, w, c)
            image = image.transpose(1, 2, 0)
            label = label.transpose(1, 2, 0)

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

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
