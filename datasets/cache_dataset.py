import gzip
import os
import pickle
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.util import CLASS2IND, CLASSES

from PIL import Image, ImageDraw
import random
import cv2
import json

class CacheDataset(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.transforms = transforms
        self.labelnames = np.array(glob(os.path.join(config.label_dir, "*", "*.json")))
        self.filenames = np.array(glob(os.path.join(config.image_dir, "*", "*.png")))


        if self.is_train:
            self.data_dir = config.train_cache_data_dir
            self._filename = glob(os.path.join(self.data_dir, "*.pkl"))
        else:
            self.data_dir = config.valid_cache_data_dir
            self._filename = glob(os.path.join(self.data_dir, "*.pkl"))

        assert len(self._filename) != 0, "Please check cache data directory !"

    def __len__(self):
        return len(self._filename)

    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min

    def __getitem__(self, idx):
        with gzip.open(os.path.join(self.data_dir, self._filename[idx]), mode="rb") as f:
            image, label = pickle.load(f)

        if self.transforms is not None:
            # (c, h ,w) -> (h, w, c)
            image = np.array(image).transpose(1, 2, 0)
            label = np.array(label).transpose(1, 2, 0)

            if self.config.copy_paste.k != 0 and self.is_train:
                randoms = random.choices([i for i in range(800)], k=self.config.copy_paste.k)
                for i in randoms:
                    target_image = cv2.imread(self.filenames[i]) / 255.
                    target_label_path = self.labelnames[i]

                    with open(target_label_path, "r") as f:
                        target_annotations = json.load(f)
                    target_annotations = target_annotations["annotations"]

                    for ann in target_annotations:
                        target_c = ann["label"]
                        target_c = CLASS2IND[target_c]
                        if target_c == 19 or target_c == 20 or target_c == 25 or target_c == 26:
                            max, min = self.get_coord(ann['points'])
                            scale = (image.shape[0] / 2048)
                            max = [int(i * scale) for i in max]
                            min = [int(i * scale) for i in min]
                            range_max = int(1800. * scale)
                            range_min = int(100. * scale)
                            x = random.randint(range_min, range_max)
                            y = random.randint(range_min, range_max)
                            alpha = int(random.randint(25,50) * scale)
                            # 0. check whether generated (x,y) coordinate is in the background 
                            bone_area_x = [i for i in range(int(400 * scale), int(1600 * scale))]
                            while x in bone_area_x:
                                x = random.randint(100 * scale,1800 * scale)
                            x -= alpha      
                            # 1. create mask for new image
                            img = Image.new('L', target_image.shape[:2], 0)
                            ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                            mask = np.array(img)
                            
                            # 2. paste maskout poly to source image
                            new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                            new_image = cv2.resize(new_image, 
                                                image.shape[:2], 
                                                interpolation=cv2.INTER_AREA)
                            if image.shape[-1] == 3:
                                image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]
                            else:
                                image[y:y+max[1]-min[1], x:x+max[0]-min[0], 0] = new_image[min[1]:max[1], min[0]:max[0], 0]                        

                            # 3. update label
                            ori_label = label[..., target_c]
                            ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                            label[..., target_c] = ori_label


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
