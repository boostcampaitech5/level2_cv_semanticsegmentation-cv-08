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
from runner import train
from utils import read_json


def main(config):
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    with open(os.path.join(config.model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)

    if config.wandb.use_wandb:
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.run_name,
            config=config,
        )

    if config.smp.use_smp:
        model = getattr(smp, config.smp.model)(
            encoder_name=config.smp.encoder_name,
            encoder_weights=config.smp.encoder_weights,
            in_channels=3,
            classes=config.num_classes,
        )
    else:
        model = getattr(models, config.model)(config.num_classes)
    model.cuda()

    optimizer = partial(getattr(torch.optim, config.optimizer.type))
    optimizer = optimizer(model.parameters(), **config.optimizer.parameters)
    criterion = getattr(loss, config.loss)()

    if config.augmentations and config.dataset != "CacheDataset":
        train_aug = getattr(augmentations, config.augmentations.name)(
            **config.augmentations.parameters
        )
        valid_aug = getattr(augmentations, "base_augmentation")(
            config.augmentations.parameters.resize, mean=0.13189, std=0.17733
        )
    else:  # config.augmentation이 false일 경우 기본 데이터셋이면 config.input_size에 맞게 resize, pickle 데이터셋으면 변환없이 그대로 입력
        train_aug = None
        valid_aug = None

    train_dataset = getattr(datasets, config.dataset)(config, is_train=True, transforms=train_aug)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valid_dataset = getattr(datasets, config.dataset)(config, is_train=False, transforms=valid_aug)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    train(config, model, train_loader, valid_loader, criterion, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path (default: None)"
    )
    args = parser.parse_args()
    config = read_json(args.config)
    main(config)
