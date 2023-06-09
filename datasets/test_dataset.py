# python native
import os
import numpy as np

# external library
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset

class XRayInferenceDataset(Dataset):
    def __init__(self, config):
        
        self.IMAGE_ROOT = config.test_image_root
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.IMAGE_ROOT)
            for root, _dirs, files in os.walk(self.IMAGE_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = A.Compose([
            A.Resize(config.input_size, config.input_size)
        ], p=1.0)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        image = torch.from_numpy(image).float()
        return image, image_name
   