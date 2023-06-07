# python native
import os
import yaml
from glob import glob
from argparse import ArgumentParser

# external library
import albumentations as A
import segmentation_models_pytorch as smp

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# utils
from dataset import XRayDataset
from runner import train
from utils import AttributeDict, set_seed


def main(args):
    # Load Data
    pngs = glob(os.path.join(args.image_dir, '*', '*.png'))
    jsons = glob(os.path.join(args.label_dir, '*', '*.json'))
    
    jsons_fn_prefix = {os.path.splitext(os.path.basename(fname))[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(os.path.basename(fname))[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)
    
    # Augmentation
    tf = A.Resize(512, 512)
    
    # Dataset
    train_dataset = XRayDataset(args, pngs, jsons, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(args, pngs, jsons, is_train=False, transforms=tf)

    # Dataloader
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    model = smp.Unet(
        encoder_name=args.encoder_name, # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=29,                     # model output channels (number of classes in your dataset)
    )
    
    # Loss function 정의
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    train(args, model, train_loader, valid_loader, criterion, optimizer)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-c", "--config",
        default="./config.yaml",
        type=str,
        help="config file path (default: ./config.yaml)",
    )

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return AttributeDict(config)


if __name__ == '__main__':
    args = parse_args()
    
    # set seed
    set_seed(args.seed)
    
    # check save model dir
    os.makedirs(args.save_model_dir, exist_ok=True)
    
    main(args)