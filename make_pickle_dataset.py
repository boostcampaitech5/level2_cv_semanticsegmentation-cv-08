import argparse
import gzip
import os
import pickle

from tqdm import tqdm

import augmentations
from datasets import XRayDataset, XRayDatasetV2
from utils import read_json


def main(config, args):
    if not os.path.exists(args.target):
        os.makedirs(os.path.join(args.target, "train"))
        os.makedirs(os.path.join(args.target, "valid"))

    train_path = os.path.join(args.target, "train")
    valid_path = os.path.join(args.target, "valid")
    
    if config.augmentations:
        train_aug = getattr(augmentations, config.augmentations.name)(
            **config.augmentations.parameters
        )
        valid_aug = getattr(augmentations, "base_augmentation")(
            config.augmentations.parameters.resize, mean=0.13189, std=0.17733
        )
    else:  # config.augmentation이 false일 경우 기본 데이터셋이면 config.input_size에 맞게 resize, pickle 데이터셋으면 변환없이 그대로 입력
        train_aug = None
        valid_aug = None

    if args.v2:
        train_dataset = XRayDatasetV2(config, transforms=train_aug)
        valid_dataset = XRayDatasetV2(config, is_train=False, transforms=valid_aug)
    else:
        train_dataset = XRayDataset(config, transforms=train_aug)
        valid_dataset = XRayDataset(config, is_train=False, transforms=valid_aug)

    for i in tqdm(range(len(train_dataset))):
        g = train_dataset[i]
        with gzip.open(os.path.join(train_path, f"{i}.pkl"), mode="wb") as f:
            pickle.dump(g, f)

    for i in tqdm(range(len(valid_dataset))):
        g = valid_dataset[i]
        with gzip.open(os.path.join(valid_path, f"{i}.pkl"), mode="wb") as f:
            pickle.dump(g, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./config.json", type=str, help="config file path (default: None)"
    )
    parser.add_argument(
        "-t",
        "--target",
        default="/opt/ml/cache_data",
        type=str,
        help="cache dataset root directory",
    )
    parser.add_argument('--v2', action='store_true', help='Use XRayDatasetV2')

    args = parser.parse_args()
    config = read_json(args.config)
    
    main(config, args)
