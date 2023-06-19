# python native
import json
import os
from glob import glob

# external library
import cv2
import numpy as np

# torch
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

# utils
from utils.util import CLASS2IND, CLASSES
from PIL import Image, ImageDraw
import random

class XRayDataset(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        self.config = config

        # Load Data
        pngs = glob(os.path.join(config.image_dir, "*", "*.png"))
        npys = glob(os.path.join(config.label_dir, "*", "*.json"))

        self.pngs = sorted(pngs)
        self.npys = sorted(npys)

        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.npys)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.

        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)
    
    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min    

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.config.image_dir, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labelnames[item]
        label_path = os.path.join(self.config.label_dir, label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            # CLASS2IND
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)

            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

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

                        bone_area_x = [i for i in range(int(400 * scale), int(1600 * scale))]
                        while x in bone_area_x:
                            x = random.randint(100 * scale,1800 * scale)
                        x -= alpha      

                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        new_image = cv2.resize(new_image, 
                                            image.shape[:2], 
                                            interpolation=cv2.INTER_AREA)
                        if image.shape[-1] == 3:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]
                        else:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], 0] = new_image[min[1]:max[1], min[0]:max[0], 0]                        

                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.transforms is not None:
            # inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)

            image = result["image"]
            # label = result["mask"] if self.is_train else label
            label = result["mask"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayDatasetV2(Dataset):
    """Loads data paths from a json file"""

    def __init__(self, config, is_train=True, transforms=None):
        self.image_dir = config.image_dir
        self.label_dir = config.label_dir
        self.transform = transforms
        self.labelnames = np.array(glob(os.path.join(config.label_dir, "*", "*.json")))
        self.filenames = np.array(glob(os.path.join(config.image_dir, "*", "*.png")))

        if is_train:
            self.json_path = config.train_json_path
        else:
            self.json_path = config.valid_json_path

        with open(self.json_path, "r") as f:
            _fnames = json.load(f)

        _fnames = dict(sorted(_fnames.items()))
        self.ids, self.fnames = [], []
        for k, v in _fnames.items():
            self.ids.append(k)
            self.fnames.extend(v)

    def __len__(self):
        return len(self.fnames)

    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min

    def __getitem__(self, idx):
        fname = self.fnames[idx]  # ex) ID002/image1661144246917
        image_path = os.path.join(self.image_dir, f"{fname}.png")
        label_path = os.path.join(self.label_dir, f"{fname}.json")

        image = cv2.imread(image_path)
        image = image / 255.0

        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        for ann in annotations:
            c = ann["label"]
            # CLASS2IND
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)

            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

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

                        bone_area_x = [i for i in range(int(400 * scale), int(1600 * scale))]
                        while x in bone_area_x:
                            x = random.randint(100 * scale,1800 * scale)
                        x -= alpha      

                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        new_image = cv2.resize(new_image, 
                                            image.shape[:2], 
                                            interpolation=cv2.INTER_AREA)
                        if image.shape[-1] == 3:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]
                        else:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], 0] = new_image[min[1]:max[1], min[0]:max[0], 0]                        

                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayDatasetFast(Dataset):
    def __init__(self, config, is_train=True, transforms=None):
        self.config = config
        self.is_train = is_train
        self.transforms = transforms

        # Load Data
        pngs = glob(os.path.join(config.image_dir, "*", "*.png"))
        npys = glob(os.path.join(config.label_dir, "*", "*.npy"))

        self.pngs = sorted(pngs)
        self.npys = sorted(npys)

        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.npys)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.

        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue

                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        self.filenames = filenames
        self.labelnames = labelnames

    def __len__(self):
        return len(self.filenames)

    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.config.image_dir, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labelnames[item]
        label_path = os.path.join(self.config.label_dir, label_name)

        # process a label of shape (H, W, NC)
        with open(label_path, "rb") as f:
            label_np = np.load(f)
            if isinstance(label_np, np.lib.npyio.NpzFile):
                label = np.load(f)["arr_0"]
            else:
                label = label_np
        label = np.unpackbits(label).reshape(2048, 2048, 29)

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

                        bone_area_x = [i for i in range(int(400 * scale), int(1600 * scale))]
                        while x in bone_area_x:
                            x = random.randint(100 * scale,1800 * scale)
                        x -= alpha      

                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        new_image = cv2.resize(new_image, 
                                            image.shape[:2], 
                                            interpolation=cv2.INTER_AREA)
                        if image.shape[-1] == 3:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]
                        else:
                            image[y:y+max[1]-min[1], x:x+max[0]-min[0], 0] = new_image[min[1]:max[1], min[0]:max[0], 0]                        

                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
