# python native
import os
from argparse import ArgumentParser

import albumentations as A
import segmentation_models_pytorch as smp

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# external library
import wandb
import yaml
from torch.utils.data import DataLoader

# utils
import loss
import datasets
from runner.train_runner import train
from utils.util import CLASSES, AttributeDict, check_directory, set_seed


def main(args):
    # Augmentation
    train_tf = A.Compose([   
            A.CenterCrop(300, 300, p=0.5,),
            A.Resize(512, 512),
            # A.HorizontalFlip(p=0.5),
        ])
    valid_tf = A.Compose([A.Resize(512, 512)])

    train_dataset = getattr(datasets, args.dataset.use)(args, is_train=True, transforms=None)
    valid_dataset = getattr(datasets, args.dataset.use)(args, is_train=False, transforms=None)

    # Dataloader
    train_loader = DataLoader( 
        dataset=train_dataset,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.valid.batch_size,
        shuffle=False,
        num_workers=args.valid.num_workers,
        drop_last=False
    )

    # Model Define
    model = getattr(smp, args.model)(
        encoder_name=args.encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=args.encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=len(CLASSES),  # model output channels (number of classes in your dataset)
    )

    if args.resume:
        print(f"Load {args.save_model_fname} weights")
        model.load_state_dict(
            torch.load(os.path.join(args.save_model_dir, args.save_model_fname)).state_dict()
        )

    # Loss function 정의
    criterion = getattr(loss, args.loss)()

    # Optimizer 정의
    optimizer = getattr(optim, args.optimizer.optim)(
        params=model.parameters(), lr=args.optimizer.learning_rate, weight_decay=args.optimizer.weight_decay
    )

    train(args, model, train_loader, valid_loader, criterion, optimizer)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default="./config.yaml",
        type=str,
        help="config file path (default: ./config.yaml)",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return AttributeDict(config)


if __name__ == "__main__":
    args = parse_args()

    # set seed
    set_seed(args.seed)

    # check save model dir
    args.inference = False
    args = check_directory(args)

    # wandb
    if args.wandb.use:
        wandb.init(
            entity=args.wandb.entity,
            project=args.wandb.project,
            name=f"{args.model}_{args.encoder_name}_{args.model_info}",
            config=args,
        )

    main(args)
