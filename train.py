import argparse
import json
import os
from functools import partial

import segmentation_models_pytorch as smp
import torch
import wandb
from torch.utils.data import DataLoader

import augmentations
import datasets
import loss
import models

from models.encoder.swin_encoder import register_encoder
from runner import train
from utils import CLASSES, CosineAnnealingWarmUpRestarts, read_json


def main(config):
    os.makedirs(config.save_model_dir, exist_ok=True)
    print(f"Save Model Directory : {config.save_model_dir}")

    with open(os.path.join(config.save_model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)

    # Wandb Connect
    if config.wandb.use:
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.run_name,
            config=config,
        )

    # Model Define
    if config.base.use == "smp":
        register_encoder()
        
        model = getattr(smp, config.base.smp.model)(
            **config.base.smp.parameters,
            in_channels=3,
            classes=len(CLASSES),
        )
    else:
        model = getattr(models, config.base.pytorch.model)(len(CLASSES))

    model.cuda()

    # Optimizer 정의
    optimizer = partial(getattr(torch.optim, config.optimizer.type))
    optimizer = optimizer(model.parameters(), **config.optimizer.parameters)

    # Loss function 정의
    criterion = getattr(loss, config.criterion)()

    # Learning Rate Scheduler 정의
    if config.scheduler.type == "CosineAnnealingWarmUpRestarts":
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, **config.scheduler.parameters)
    else:
        lr_scheduler = partial(getattr(torch.optim.lr_scheduler, config.scheduler.type))(
            optimizer, **config.scheduler.parameters
        )

    # 학습된 Model Info Load -> path를 입력해주세요
    if config.resume_from:
        print(f"Load {config.resume_from}")
        if os.path.splitext(config.resume_from)[1] == ".pt":
            model = torch.load(config.resume_from)
        else:
            checkpoint = torch.load(os.path.join(config.save_model_dir, config.model_file_name))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Augmentation 정의
    train_aug = getattr(augmentations, config.train.augmentations.name)(
        **config.train.augmentations.parameters
    )
    valid_aug = getattr(augmentations, config.valid.augmentations.name)(
        **config.valid.augmentations.parameters
    )

    train_dataset = getattr(datasets, config.dataset)(config, is_train=True, transforms=train_aug)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=True,
    )

    valid_dataset = getattr(datasets, config.dataset)(config, is_train=False, transforms=valid_aug)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid.batch_size,
        shuffle=False,
        num_workers=config.valid.num_workers,
        drop_last=False,
    )

    train(config, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path (default: None)"
    )
    args = parser.parse_args()
    config = read_json(args.config)
    main(config)
