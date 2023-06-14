import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import augmentations
from datasets import XRayDataset, XRayDatasetV2
from utils import read_json


def main(config, args):
    if not os.path.exists(args.target):
        os.makedirs(os.path.join(args.target, "train"))
        os.makedirs(os.path.join(args.target, "valid"))

    train_path = os.path.join(args.target, "train")
    valid_path = os.path.join(args.target, "valid")

    train_aug = getattr(augmentations, config.train_augmentations.name)(
        **config.train_augmentations.parameters
    )
    valid_aug = getattr(augmentations, config.valid_augmentations.name)(
        **config.valid_augmentations.parameters
    )

    if args.v2:
        train_dataset = XRayDatasetV2(config, is_train=True, transforms=train_aug)
        valid_dataset = XRayDatasetV2(config, is_train=False, transforms=valid_aug)
    else:
        train_dataset = XRayDataset(config, is_train=True, transforms=train_aug)
        valid_dataset = XRayDataset(config, is_train=False, transforms=valid_aug)

    train_hdf5 = os.path.join(train_path, f"train_uint8.h5py")
    valid_hdf5 = os.path.join(valid_path, f"valid_uint8.h5py")

    with h5py.File(train_hdf5, "w", rdcc_nslots=11213, rdcc_nbytes=1024**3, rdcc_w0=1) as hf:
        for idx, (image, label) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            # float32 -> uint8
            image = np.array(image)
            image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

            label = np.array(label)
            label = cv2.normalize(label, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

            iset = hf.create_dataset(
                f"{idx}/image",
                data=image,
                shape=image.shape,
                dtype="uint8",
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

            lset = hf.create_dataset(
                f"{idx}/label",
                data=label,
                shape=label.shape,
                dtype="uint8",
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

    with h5py.File(valid_hdf5, "w", rdcc_nslots=11213, rdcc_nbytes=1024**3, rdcc_w0=1) as hf:
        for idx, (image, label) in tqdm(enumerate(valid_dataset), total=len(valid_dataset)):
            # float32 -> uint8
            image = np.array(image)
            image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

            label = np.array(label)
            label = cv2.normalize(label, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

            iset = hf.create_dataset(
                f"{idx}/image",
                data=image,
                shape=image.shape,
                dtype="uint8",
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

            lset = hf.create_dataset(
                f"{idx}/label",
                data=label,
                shape=label.shape,
                dtype="uint8",
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="../config.json",
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="/opt/ml/input/data_hdf5",
        type=str,
        help="cache dataset root directory",
    )
    parser.add_argument("--v2", action="store_true", help="Use XRayDatasetV2")

    args = parser.parse_args()
    config = read_json(args.config)

    main(config, args)
