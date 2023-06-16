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

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.config.image_dir, image_name)

        image = cv2.imread(image_path)
        if self.config.gray and image.shape[-1] == 3:
                image = image[..., 0, np.newaxis]
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


        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]  # ex) ID002/image1661144246917
        image_path = os.path.join(self.image_dir, f"{fname}.png")
        label_path = os.path.join(self.label_dir, f"{fname}.json")

        image = cv2.imread(image_path)
        if self.config.gray and image.shape[-1] == 3:
                image = image[..., 0, np.newaxis]      
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
        self.is_train = is_train

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.config.image_dir, image_name)

        image = cv2.imread(image_path)
        if self.config.gray and image.shape[-1] == 3:
                image = image[..., 0, np.newaxis]
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
