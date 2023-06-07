from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import GroupKFold
import os
import cv2
import json
import torch
import argparse
from parse_config import ConfigParser
import collections
import albumentations as A

class XRayDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        
        #set -> list(using sort) -> numpy
        #이 부분이 시간이 오래걸릴 것 같은데 최적화하는 방법이 없을까??
        #line number 19 to 37
        #모든 json file들을 모으는 것이다.
        self.jsons = {
            os.path.relpath(os.path.join(root, fname), start=config['label_root'])
            for root, _dirs, files in os.walk(config['label_root'])
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        self.pngs = {
            os.path.relpath(os.path.join(root, fname), start=config['image_root'])
            for root, _dirs, files in os.walk(config['image_root'])
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        self.pngs = sorted(self.pngs)
        self.jsons = sorted(self.jsons)
        
        _filenames = np.array(self.pngs)
        _labelnames = np.array(self.jsons)
        
        self.CLASSES=[
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        
        self.CLASS2IND = {v:i for i,v in enumerate(self.CLASSES)}
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
        
        self.transforms = A.Compose([
            A.Resize(512, 512)
        ], p=1.0)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.config['image_root'], image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.config['label_root'], label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(self.CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            #CLASS2IND
            class_ind = self.CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label
    
    
class XRayInferenceDataset(Dataset):
    def __init__(self, config):
        self.CLASSES=[
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        
        self.IMAGE_ROOT = config['test_image_root']
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
            A.Resize(512, 512)
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
    
if __name__=="__main__":    
    args = argparse.ArgumentParser(description='Segmentation Template')
    args.add_argument('-c', '--config', default=None, type=str,
                    help='config file path (default: None)')
    config = ConfigParser.from_args(args)
    
    tf = A.Resize(512, 512)
    
    train_dataset = XRayDataset(config, is_train=True)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = config['train_batch_size'],
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    
    for idx, (images, masks) in enumerate(train_loader):
        print(images.shape)
        print(masks.shape)
        break